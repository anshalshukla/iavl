use prost::encoding as prost_encoding;
use zigzag::ZigZag;

// Use the official ICS23 crate
use ics23::{
    CommitmentProof, ExistenceProof, HashOp, HostFunctionsManager, InnerOp, LeafOp, LengthOp,
    NonExistenceProof, iavl_spec, verify_membership, verify_non_membership,
};

use crate::{
    immutable_tree::ImmutableTree,
    node_db::KVStoreWithBatch,
    proof::{PathToLeaf, ProofError},
    types::{BoundedUintTrait, U7, U63},
};

const LENGTH_PREFIX: u8 = 0x20;

/// Trait for ICS23 proof operations using the official ics23 crate
pub trait ICS23Proof<DB: KVStoreWithBatch> {
    /// Get membership proof for a key that exists in the tree
    fn get_membership_proof(&self, key: &[u8]) -> Result<CommitmentProof, ProofError>;

    /// Verify membership proof
    fn verify_membership(&self, proof: &CommitmentProof, key: &[u8]) -> Result<bool, ProofError>;

    /// Get non-membership proof for a key that doesn't exist in the tree
    fn get_non_membership_proof(&self, key: &[u8]) -> Result<CommitmentProof, ProofError>;

    /// Verify non-membership proof
    fn verify_non_membership(
        &self,
        proof: &CommitmentProof,
        key: &[u8],
    ) -> Result<bool, ProofError>;

    /// Get proof (either membership or non-membership)
    fn get_proof(&self, key: &[u8]) -> Result<CommitmentProof, ProofError>;

    /// Verify proof (either membership or non-membership)
    fn verify_proof(&self, proof: &CommitmentProof, key: &[u8]) -> Result<bool, ProofError>;

    /// Create existence proof for a key
    fn create_existence_proof(&self, key: &[u8]) -> Result<ExistenceProof, ProofError>;
}

impl<DB: KVStoreWithBatch> ICS23Proof<DB> for ImmutableTree<DB> {
    fn get_membership_proof(&self, key: &[u8]) -> Result<CommitmentProof, ProofError> {
        let exist = self.create_existence_proof(key)?;
        Ok(CommitmentProof {
            proof: Some(ics23::commitment_proof::Proof::Exist(exist)),
        })
    }

    fn verify_membership(&self, proof: &CommitmentProof, key: &[u8]) -> Result<bool, ProofError> {
        let val = self.get(key)?;
        if val.is_none() {
            return Ok(false);
        }

        let root = self.hash()?;
        let spec = iavl_spec();
        Ok(verify_membership::<HostFunctionsManager>(
            proof,
            &spec,
            &root,
            key,
            &val.unwrap(),
        ))
    }

    fn get_non_membership_proof(&self, key: &[u8]) -> Result<CommitmentProof, ProofError> {
        // Get index and value for the key
        let (idx, val) = self.get_with_index(key)?;

        if val.is_some() {
            return Err(ProofError::KeyExistsInState);
        }

        let mut nonexist = NonExistenceProof {
            key: key.to_vec(),
            left: None,
            right: None,
        };

        // Get left neighbor if exists
        if idx >= 1 {
            if let Ok((leftkey, _)) = self.get_by_index(idx - 1) {
                nonexist.left = Some(self.create_existence_proof(&leftkey)?);
            }
        }

        // Get right neighbor if exists
        if let Ok((rightkey, _)) = self.get_by_index(idx) {
            nonexist.right = Some(self.create_existence_proof(&rightkey)?);
        }

        Ok(CommitmentProof {
            proof: Some(ics23::commitment_proof::Proof::Nonexist(nonexist)),
        })
    }

    fn verify_non_membership(
        &self,
        proof: &CommitmentProof,
        key: &[u8],
    ) -> Result<bool, ProofError> {
        let root = self.hash()?;
        let spec = iavl_spec();
        Ok(verify_non_membership::<HostFunctionsManager>(
            proof, &spec, &root, key,
        ))
    }

    fn get_proof(&self, key: &[u8]) -> Result<CommitmentProof, ProofError> {
        let exists = self.has(&key.to_vec())?;

        if exists {
            self.get_membership_proof(key)
        } else {
            self.get_non_membership_proof(key)
        }
    }

    fn verify_proof(&self, proof: &CommitmentProof, key: &[u8]) -> Result<bool, ProofError> {
        match &proof.proof {
            Some(ics23::commitment_proof::Proof::Exist(_)) => self.verify_membership(proof, key),
            Some(ics23::commitment_proof::Proof::Nonexist(_)) => {
                self.verify_non_membership(proof, key)
            }
            _ => Ok(false),
        }
    }

    fn create_existence_proof(&self, key: &[u8]) -> Result<ExistenceProof, ProofError> {
        // Get path to leaf
        let (path, node_opt) = self.root.read()?.path_to_leaf(self, key, self.version)?;
        let node = node_opt.ok_or(ProofError::NodeNotFound)?;
        let node_guard = node.read()?;

        // Determine node version
        let node_version = if let Some(ref node_key) = node_guard.node_key {
            node_key.version
        } else {
            U63::new(self.version.get() + 1).map_err(|_| {
                ProofError::TypesError(crate::types::BoundedUintError::ExceedsSignedInteger)
            })?
        };

        let value = node_guard.value.as_ref().ok_or(ProofError::NoValue)?;

        Ok(ExistenceProof {
            key: node_guard.key.clone(),
            value: value.clone(),
            leaf: Some(convert_leaf_op(node_version.get())),
            path: convert_inner_ops(path),
        })
    }
}

// we cannot get the proofInnerNode type, so we need to do the whole path in one function
fn convert_inner_ops(path: PathToLeaf) -> Vec<InnerOp> {
    let mut inner_ops = Vec::new();
    for (i, node_proof) in path.path.iter().rev().enumerate() {
        if node_proof.left.is_some() {
            inner_ops.push(convert_inner_op_left(
                node_proof.height,
                node_proof.size,
                node_proof.version,
                node_proof.left.as_ref().unwrap(),
            ));
        } else {
            inner_ops.push(convert_inner_op_right(
                node_proof.height,
                node_proof.size,
                node_proof.version,
                node_proof.right.as_ref().unwrap(),
            ));
        }
    }

    inner_ops
}

/// Convert version to LeafOp using ICS23 standard
fn convert_leaf_op(version: u64) -> LeafOp {
    // Create prefix based on IAVL leaf node structure
    let mut prefix = Vec::new();

    // Height = 0 (zigzag encoded)
    let height = ZigZag::encode(0i8);
    prost_encoding::encode_varint(height.into(), &mut prefix);

    // Size = 1 (zigzag encoded)
    let size = ZigZag::encode(1i64);
    prost_encoding::encode_varint(size, &mut prefix);

    // Version (zigzag encoded)
    let version_encoded = ZigZag::encode(version as i64);
    prost_encoding::encode_varint(version_encoded, &mut prefix);

    LeafOp {
        hash: HashOp::Sha256.into(),
        prehash_key: HashOp::NoHash.into(),
        prehash_value: HashOp::Sha256.into(),
        length: LengthOp::VarProto.into(),
        prefix,
    }
}

/// Convert to InnerOp when going left (left hash is stored)
fn convert_inner_op_left(height: U7, size: U63, version: U63, left_hash: &Vec<u8>) -> InnerOp {
    let mut prefix = Vec::new();

    // Encode height, size, version
    let height_encoded = ZigZag::encode(height.as_signed());
    prost_encoding::encode_varint(height_encoded.into(), &mut prefix);

    let size_encoded = ZigZag::encode(size.as_signed());
    prost_encoding::encode_varint(size_encoded, &mut prefix);

    let version_encoded = ZigZag::encode(version.as_signed());
    prost_encoding::encode_varint(version_encoded, &mut prefix);

    // Child goes on left, sibling hash on right
    prefix.push(LENGTH_PREFIX); // 32 bytes for child hash

    prefix.extend_from_slice(left_hash);

    prefix.push(LENGTH_PREFIX); // 32 bytes for right hash

    InnerOp {
        hash: HashOp::Sha256.into(),
        prefix,
        suffix: Vec::new(),
    }
}

/// Convert to InnerOp when going right (right hash is stored)
fn convert_inner_op_right(height: U7, size: U63, version: U63, right_hash: &Vec<u8>) -> InnerOp {
    let mut prefix = Vec::new();

    // Encode height, size, version
    let height_encoded = ZigZag::encode(height.as_signed());
    prost_encoding::encode_varint(height_encoded.into(), &mut prefix);

    let size_encoded = ZigZag::encode(size.as_signed());
    prost_encoding::encode_varint(size_encoded, &mut prefix);

    let version_encoded = ZigZag::encode(version.as_signed());
    prost_encoding::encode_varint(version_encoded, &mut prefix);

    // Left hash first, then child
    prefix.push(LENGTH_PREFIX); // 32 bytes for left hash

    let mut suffix = Vec::new();

    suffix.push(LENGTH_PREFIX); // 32 bytes for child hash
    suffix.extend_from_slice(right_hash);

    InnerOp {
        hash: HashOp::Sha256.into(),
        prefix,
        suffix,
    }
}
