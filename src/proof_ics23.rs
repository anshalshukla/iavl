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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestUtils;
    use crate::types::BoundedUintTrait;
    use rand::Rng;

    #[test]
    fn test_get_membership_small_tree() {
        // Create a random IAVL tree with height 3 (small tree)
        let tree = TestUtils::create_random_iavl_tree(2).expect("Failed to create random tree");

        // Collect all keys from the tree to select one randomly
        let mut all_keys = Vec::new();
        let mut iterator = tree
            .clone()
            .iterator(&[], &[], true)
            .expect("Failed to create iterator");

        while iterator.valid() {
            all_keys.push(iterator.key().clone());
            iterator.next();
        }

        if all_keys.is_empty() {
            println!("No keys in tree, skipping membership test");
            return;
        }

        // Select a random key from the tree
        let mut rng = rand::thread_rng();
        let random_index = rng.gen_range(0..all_keys.len());
        let selected_key = &all_keys[random_index];

        // Get the value for the selected key
        let value = tree
            .get(selected_key)
            .expect("Failed to get value")
            .expect("Key should exist");

        // Create membership proof
        let proof = tree
            .get_membership_proof(selected_key)
            .expect("Failed to create membership proof");

        // Verify the membership proof (using correct method signature)
        let is_valid = tree
            .verify_membership(&proof, selected_key)
            .expect("Failed to verify membership");

        assert!(is_valid, "Membership proof should be valid");

        println!(
            "Successfully verified membership proof for key: {:?}",
            selected_key
        );
    }

    #[test]
    fn test_get_membership_big_tree() {
        // Create a random IAVL tree with height 4 (bigger tree)
        let tree = TestUtils::create_random_iavl_tree(4).expect("Failed to create random tree");

        // Collect all keys from the tree
        let mut all_keys = Vec::new();
        let mut iterator = tree
            .clone()
            .iterator(&[], &[], true)
            .expect("Failed to create iterator");

        while iterator.valid() {
            all_keys.push(iterator.key().clone());
            iterator.next();
        }

        if all_keys.is_empty() {
            println!("No keys in tree, skipping membership test");
            return;
        }

        // Test membership for multiple random keys
        let mut rng = rand::thread_rng();
        let test_count = std::cmp::min(5, all_keys.len()); // Test up to 5 keys

        for i in 0..test_count {
            let random_index = rng.gen_range(0..all_keys.len());
            let selected_key = &all_keys[random_index];

            println!(
                "Testing membership for key {} of {}: {:?}",
                i + 1,
                test_count,
                selected_key
            );

            // Get the value for the selected key
            let value = tree
                .get(selected_key)
                .expect("Failed to get value")
                .expect("Key should exist");

            // Create membership proof
            let proof = tree
                .get_membership_proof(selected_key)
                .expect("Failed to create membership proof");

            // Verify the membership proof (using correct method signature)
            let is_valid = tree
                .verify_membership(&proof, selected_key)
                .expect("Failed to verify membership");

            assert!(
                is_valid,
                "Membership proof should be valid for key: {:?}",
                selected_key
            );
        }

        println!(
            "Successfully verified membership proofs for {} keys",
            test_count
        );
    }

    #[test]
    fn test_get_membership_different_positions() {
        // Create a random IAVL tree
        let tree = TestUtils::create_random_iavl_tree(3).expect("Failed to create random tree");

        // Collect all keys and sort them
        let mut all_keys = Vec::new();
        let mut iterator = tree
            .clone()
            .iterator(&[], &[], true)
            .expect("Failed to create iterator");

        while iterator.valid() {
            all_keys.push(iterator.key().clone());
            iterator.next();
        }

        if all_keys.len() < 3 {
            println!("Not enough keys in tree, skipping position test");
            return;
        }

        // Test leftmost, middle, and rightmost keys (similar to Go test cases)
        let positions = vec![
            ("leftmost", 0),
            ("middle", all_keys.len() / 2),
            ("rightmost", all_keys.len() - 1),
        ];

        for (position_name, index) in positions {
            let selected_key = &all_keys[index];

            println!("Testing {} key: {:?}", position_name, selected_key);

            // Get the value for the selected key
            let value = tree
                .get(selected_key)
                .expect("Failed to get value")
                .expect("Key should exist");

            // Create membership proof
            let proof = tree
                .get_membership_proof(selected_key)
                .expect("Failed to create membership proof");

            // Verify the membership proof (using correct method signature)
            let is_valid = tree
                .verify_membership(&proof, selected_key)
                .expect("Failed to verify membership");

            assert!(
                is_valid,
                "Membership proof should be valid for {} key: {:?}",
                position_name, selected_key
            );

            println!("Successfully verified {} membership proof", position_name);
        }
    }

    #[test]
    fn test_get_membership_multiple_trees() {
        // Test membership proofs for trees of different heights
        for height in 2..=4 {
            println!("Testing membership for tree height: {}", height);

            let tree =
                TestUtils::create_random_iavl_tree(height).expect("Failed to create random tree");

            // Get all keys
            let mut all_keys = Vec::new();
            let mut iterator = tree
                .clone()
                .iterator(&[], &[], true)
                .expect("Failed to create iterator");

            while iterator.valid() {
                all_keys.push(iterator.key().clone());
                iterator.next();
            }

            if all_keys.is_empty() {
                println!("No keys in tree height {}, skipping", height);
                continue;
            }

            // Test a few random keys from this tree
            let mut rng = rand::thread_rng();
            let test_count = std::cmp::min(3, all_keys.len());

            for i in 0..test_count {
                let random_index = rng.gen_range(0..all_keys.len());
                let selected_key = &all_keys[random_index];

                // Get value and create proof
                let value = tree
                    .get(selected_key)
                    .expect("Failed to get value")
                    .expect("Key should exist");
                let proof = tree
                    .get_membership_proof(selected_key)
                    .expect("Failed to create membership proof");

                // Verify proof (using correct method signature)
                let is_valid = tree
                    .verify_membership(&proof, selected_key)
                    .expect("Failed to verify membership");

                assert!(
                    is_valid,
                    "Membership proof should be valid for height {} key: {:?}",
                    height, selected_key
                );
            }

            println!(
                "Successfully verified {} membership proofs for height {}",
                test_count, height
            );
        }
    }

    #[test]
    fn test_get_membership_with_mock_trees() {
        // Test with simpler mock trees as well

        // Test single node tree
        let single_tree =
            TestUtils::create_mock_tree_with_root().expect("Failed to create single node tree");
        let key = 6u32.to_be_bytes().to_vec();
        let value = single_tree
            .get(&key)
            .expect("Failed to get value")
            .expect("Key should exist");

        let proof = single_tree
            .get_membership_proof(&key)
            .expect("Failed to create membership proof");
        // Verify using correct method signature (only proof and key)
        let is_valid = single_tree
            .verify_membership(&proof, &key)
            .expect("Failed to verify membership");

        assert!(
            is_valid,
            "Membership proof should be valid for single node tree"
        );
        println!("Successfully verified membership for single node tree");

        // Test tree with children
        let tree_with_children = TestUtils::create_mock_tree_with_root_and_children()
            .expect("Failed to create tree with children");

        // Test all three keys in the tree
        let test_keys = vec![5u32, 6u32, 7u32];

        for test_key in test_keys {
            let key_bytes = test_key.to_be_bytes().to_vec();
            let value = tree_with_children
                .get(&key_bytes)
                .expect("Failed to get value")
                .expect("Key should exist");

            let proof = tree_with_children
                .get_membership_proof(&key_bytes)
                .expect("Failed to create membership proof");
            // Verify using correct method signature (only proof and key)
            let is_valid = tree_with_children
                .verify_membership(&proof, &key_bytes)
                .expect("Failed to verify membership");

            assert!(
                is_valid,
                "Membership proof should be valid for key: {}",
                test_key
            );
        }

        println!("Successfully verified membership for tree with children");
    }

    #[test]
    fn test_get_non_membership_proof() {
        // Create a random IAVL tree
        let tree = TestUtils::create_random_iavl_tree(3).expect("Failed to create random tree");

        // Test with a key that definitely doesn't exist
        let non_existent_key = 999999u32.to_be_bytes().to_vec();

        // Verify the key doesn't exist
        let exists = tree
            .has(&non_existent_key)
            .expect("Failed to check key existence");
        assert!(!exists, "Key should not exist in tree");

        // Create non-membership proof
        let proof = tree
            .get_non_membership_proof(&non_existent_key)
            .expect("Failed to create non-membership proof");

        // Verify the non-membership proof
        let is_valid = tree
            .verify_non_membership(&proof, &non_existent_key)
            .expect("Failed to verify non-membership");

        assert!(is_valid, "Non-membership proof should be valid");

        println!(
            "Successfully verified non-membership proof for key: {:?}",
            non_existent_key
        );
    }

    #[test]
    fn test_get_proof_automatic() {
        // Create a random IAVL tree
        let tree = TestUtils::create_random_iavl_tree(3).expect("Failed to create random tree");

        // Get one existing key
        let mut iterator = tree
            .clone()
            .iterator(&[], &[], true)
            .expect("Failed to create iterator");

        if !iterator.valid() {
            println!("No keys in tree, skipping automatic proof test");
            return;
        }

        let existing_key = iterator.key().clone();
        let non_existent_key = 999999u32.to_be_bytes().to_vec();

        // Test automatic proof for existing key (should be membership proof)
        let membership_proof = tree
            .get_proof(&existing_key)
            .expect("Failed to get proof for existing key");
        let is_membership_valid = tree
            .verify_proof(&membership_proof, &existing_key)
            .expect("Failed to verify membership proof");
        assert!(
            is_membership_valid,
            "Automatic membership proof should be valid"
        );

        // Test automatic proof for non-existent key (should be non-membership proof)
        let non_membership_proof = tree
            .get_proof(&non_existent_key)
            .expect("Failed to get proof for non-existent key");
        let is_non_membership_valid = tree
            .verify_proof(&non_membership_proof, &non_existent_key)
            .expect("Failed to verify non-membership proof");
        assert!(
            is_non_membership_valid,
            "Automatic non-membership proof should be valid"
        );

        println!("Successfully verified automatic proof selection");
    }
}
