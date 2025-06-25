use core::{cmp, mem};

use crate::{
    kvstore::KVStore,
    node::{ArlockNode, DraftedNode, db::NodeDb, info::Drafted},
    types::{U7, U63},
};

use super::{Child, InnerNode, InnerNodeError, Result};

impl InnerNode<Drafted> {
    // TODO: make it simpler and concise, and devise a strategy to avoid returning new node
    pub fn make_balanced<DB>(&mut self, ndb: &NodeDb<DB>) -> Result<Option<Self>>
    where
        DB: KVStore,
    {
        let extract_full = |child: &mut Child| -> Result<_> {
            let node = match child.extract()? {
                Child::Full(full) => full,
                Child::Part(nk) => ndb
                    .fetch_one_node(&nk)?
                    .ok_or(InnerNodeError::ChildNotFound)?
                    .into(),
            };

            Ok(node)
        };

        let height_size_pair = |node: &ArlockNode| -> Result<_> {
            let node_r = node.read()?;
            Ok((node_r.height(), node_r.size()))
        };

        let left = extract_full(self.left_mut())?;
        let right = extract_full(self.right_mut())?;

        let (left_height, left_size) = height_size_pair(&left)?;
        let (right_height, right_size) = height_size_pair(&right)?;

        let diff = left_height.to_signed() - right_height.to_signed();

        if (-1..=1).contains(&diff) {
            *self.left_mut() = Child::Full(left);
            *self.right_mut() = Child::Full(right);

            return Ok(None);
        }

        if diff > 1 {
            let mut lw = left.write()?;

            // unwrap is safe because left must be inner when diff > 1
            let ll = lw.left_mut().map(extract_full).transpose()?.unwrap();
            let lr = lw.right_mut().map(extract_full).transpose()?.unwrap();

            let (ll_height, ll_size) = height_size_pair(&ll)?;
            let (lr_height, lr_size) = height_size_pair(&lr)?;

            let left_diff = ll_height.to_signed() - lr_height.to_signed();

            if left_diff >= 0 {
                // left-left case: one right rotation on self.

                let new_right = {
                    // TODO: ascertain whether 1 can be directly added without overflow checks
                    let new_right_height = cmp::max(right_height, lr_height)
                        .get()
                        .checked_add(1)
                        .and_then(U7::new)
                        .ok_or(InnerNodeError::Overflow)?;

                    // TODO: ascertain whether additions can be direct without overflow checks
                    let new_right_size = right_size
                        .get()
                        .checked_add(lr_size.get())
                        .ok_or(InnerNodeError::Overflow)?
                        .checked_add(1)
                        .and_then(U63::new)
                        .ok_or(InnerNodeError::Overflow)?;

                    InnerNode::builder()
                        .key(self.key().clone())
                        .height(new_right_height)
                        .size(new_right_size)
                        .left(Child::Full(lr))
                        .right(Child::Full(right))
                        .build()
                };

                let new_root = {
                    // TODO: ascertain whether 1 can be directly added without overflow checks
                    let new_root_height = cmp::max(ll_height, new_right.height())
                        .get()
                        .checked_add(1)
                        .and_then(U7::new)
                        .ok_or(InnerNodeError::Overflow)?;

                    // TODO: ascertain whether additions can be direct without overflow checks
                    let new_root_size = ll_size
                        .get()
                        .checked_add(new_right.size().get())
                        .ok_or(InnerNodeError::Overflow)?
                        .checked_add(1)
                        .and_then(U63::new)
                        .ok_or(InnerNodeError::Overflow)?;

                    InnerNode::builder()
                        .key(lw.key().clone())
                        .height(new_root_height)
                        .size(new_root_size)
                        .left(Child::Full(ll))
                        .right(Child::Full(DraftedNode::from(new_right).into()))
                        .build()
                };

                return Ok(Some(mem::replace(self, new_root)));
            }

            // left-right case: one left rotation on left, and then one right rotation on self

            let mut lrw = lr.write()?;

            let lrl = lrw.left_mut().map(extract_full).transpose()?.unwrap();
            let lrr = lrw.right_mut().map(extract_full).transpose()?.unwrap();

            let (lrl_height, lrl_size) = height_size_pair(&lrl)?;
            let (lrr_height, lrr_size) = height_size_pair(&lrr)?;

            let new_left = {
                // TODO: ascertain whether 1 can be directly added without overflow checks
                let new_left_height = cmp::max(ll_height, lrl_height)
                    .get()
                    .checked_add(1)
                    .and_then(U7::new)
                    .ok_or(InnerNodeError::Overflow)?;

                // TODO: ascertain whether additions can be direct without overflow checks
                let new_left_size = ll_size
                    .get()
                    .checked_add(lrl_size.get())
                    .ok_or(InnerNodeError::Overflow)?
                    .checked_add(1)
                    .and_then(U63::new)
                    .ok_or(InnerNodeError::Overflow)?;

                InnerNode::builder()
                    .key(lw.key().clone())
                    .height(new_left_height)
                    .size(new_left_size)
                    .left(Child::Full(ll))
                    .right(Child::Full(lrl))
                    .build()
            };

            let new_right = {
                // TODO: ascertain whether 1 can be directly added without overflow checks
                let new_right_height = cmp::max(lrr_height, right_height)
                    .get()
                    .checked_add(1)
                    .and_then(U7::new)
                    .ok_or(InnerNodeError::Overflow)?;

                // TODO: ascertain whether additions can be direct without overflow checks
                let new_right_size = lrr_size
                    .get()
                    .checked_add(right_size.get())
                    .ok_or(InnerNodeError::Overflow)?
                    .checked_add(1)
                    .and_then(U63::new)
                    .ok_or(InnerNodeError::Overflow)?;

                InnerNode::builder()
                    .key(self.key().clone())
                    .height(new_right_height)
                    .size(new_right_size)
                    .left(Child::Full(lrr))
                    .right(Child::Full(right))
                    .build()
            };

            let new_root = {
                // TODO: ascertain whether 1 can be directly added without overflow checks
                let new_root_height = cmp::max(new_left.height(), new_right.height())
                    .get()
                    .checked_add(1)
                    .and_then(U7::new)
                    .ok_or(InnerNodeError::Overflow)?;

                // TODO: ascertain whether additions can be direct without overflow checks
                let new_root_size = new_left
                    .size()
                    .get()
                    .checked_add(new_right.size().get())
                    .ok_or(InnerNodeError::Overflow)?
                    .checked_add(1)
                    .and_then(U63::new)
                    .ok_or(InnerNodeError::Overflow)?;

                InnerNode::builder()
                    .key(lrw.key().clone())
                    .height(new_root_height)
                    .size(new_root_size)
                    .left(Child::Full(DraftedNode::from(new_left).into()))
                    .right(Child::Full(DraftedNode::from(new_right).into()))
                    .build()
            };

            return Ok(Some(mem::replace(self, new_root)));
        }

        let mut rw = right.write()?;

        // unwrap is safe because left must be inner when diff < -1
        let rl = rw.left_mut().map(extract_full).transpose()?.unwrap();
        let rr = rw.right_mut().map(extract_full).transpose()?.unwrap();

        let (rl_height, rl_size) = height_size_pair(&rl)?;
        let (rr_height, rr_size) = height_size_pair(&rr)?;

        let right_diff = rl_height.to_signed() - rr_height.to_signed();

        if right_diff <= 0 {
            // right-right case: one left rotation on self.

            let new_left = {
                // TODO: ascertain whether 1 can be directly added without overflow checks
                let new_left_height = cmp::max(left_height, rl_height)
                    .get()
                    .checked_add(1)
                    .and_then(U7::new)
                    .ok_or(InnerNodeError::Overflow)?;

                // TODO: ascertain whether additions can be direct without overflow checks
                let new_left_size = left_size
                    .get()
                    .checked_add(rl_size.get())
                    .ok_or(InnerNodeError::Overflow)?
                    .checked_add(1)
                    .and_then(U63::new)
                    .ok_or(InnerNodeError::Overflow)?;

                InnerNode::builder()
                    .key(self.key().clone())
                    .height(new_left_height)
                    .size(new_left_size)
                    .left(Child::Full(left))
                    .right(Child::Full(rl))
                    .build()
            };

            let new_root = {
                // TODO: ascertain whether 1 can be directly added without overflow checks
                let new_root_height = cmp::max(new_left.height(), rr_height)
                    .get()
                    .checked_add(1)
                    .and_then(U7::new)
                    .ok_or(InnerNodeError::Overflow)?;

                // TODO: ascertain whether additions can be direct without overflow checks
                let new_root_size = rr_size
                    .get()
                    .checked_add(new_left.size().get())
                    .ok_or(InnerNodeError::Overflow)?
                    .checked_add(1)
                    .and_then(U63::new)
                    .ok_or(InnerNodeError::Overflow)?;

                InnerNode::builder()
                    .key(rw.key().clone())
                    .height(new_root_height)
                    .size(new_root_size)
                    .left(Child::Full(DraftedNode::from(new_left).into()))
                    .right(Child::Full(rr))
                    .build()
            };

            return Ok(Some(mem::replace(self, new_root)));
        }

        // right-left case: one right rotation on right, and then one left rotation on self

        let mut rlw = rl.write()?;

        let rll = rlw.left_mut().map(extract_full).transpose()?.unwrap();
        let rlr = rlw.right_mut().map(extract_full).transpose()?.unwrap();

        let (rll_height, rll_size) = height_size_pair(&rll)?;
        let (rlr_height, rlr_size) = height_size_pair(&rlr)?;

        let new_left = {
            // TODO: ascertain whether 1 can be directly added without overflow checks
            let new_left_height = cmp::max(left_height, rll_height)
                .get()
                .checked_add(1)
                .and_then(U7::new)
                .ok_or(InnerNodeError::Overflow)?;

            // TODO: ascertain whether additions can be direct without overflow checks
            let new_left_size = left_size
                .get()
                .checked_add(rll_size.get())
                .ok_or(InnerNodeError::Overflow)?
                .checked_add(1)
                .and_then(U63::new)
                .ok_or(InnerNodeError::Overflow)?;

            InnerNode::builder()
                .key(self.key().clone())
                .height(new_left_height)
                .size(new_left_size)
                .left(Child::Full(left))
                .right(Child::Full(rll))
                .build()
        };

        let new_right = {
            // TODO: ascertain whether 1 can be directly added without overflow checks
            let new_right_height = cmp::max(rlr_height, rr_height)
                .get()
                .checked_add(1)
                .and_then(U7::new)
                .ok_or(InnerNodeError::Overflow)?;

            // TODO: ascertain whether additions can be direct without overflow checks
            let new_right_size = rlr_size
                .get()
                .checked_add(rr_size.get())
                .ok_or(InnerNodeError::Overflow)?
                .checked_add(1)
                .and_then(U63::new)
                .ok_or(InnerNodeError::Overflow)?;

            InnerNode::builder()
                .key(rw.key().clone())
                .height(new_right_height)
                .size(new_right_size)
                .left(Child::Full(rlr))
                .right(Child::Full(rr))
                .build()
        };

        let new_root = {
            // TODO: ascertain whether 1 can be directly added without overflow checks
            let new_root_height = cmp::max(new_left.height(), new_right.height())
                .get()
                .checked_add(1)
                .and_then(U7::new)
                .ok_or(InnerNodeError::Overflow)?;

            // TODO: ascertain whether additions can be direct without overflow checks
            let new_root_size = new_left
                .size()
                .get()
                .checked_add(new_right.size().get())
                .ok_or(InnerNodeError::Overflow)?
                .checked_add(1)
                .and_then(U63::new)
                .ok_or(InnerNodeError::Overflow)?;

            InnerNode::builder()
                .key(self.key().clone())
                .height(new_root_height)
                .size(new_root_size)
                .left(Child::Full(DraftedNode::from(new_left).into()))
                .right(Child::Full(DraftedNode::from(new_right).into()))
                .build()
        };

        Ok(Some(mem::replace(self, new_root)))
    }
}
