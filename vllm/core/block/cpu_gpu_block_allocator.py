from __future__ import annotations

from typing import Dict, List, Optional

from vllm.core.block.block_table import BlockTable
from vllm.core.block.interfaces import (Block, BlockAllocator,
                                        DeviceAwareBlockAllocator)
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.utils import Device


class CpuGpuBlockAllocator(DeviceAwareBlockAllocator):
    """A block allocator that can allocate blocks on both CPU and GPU memory.

    This class implements the `DeviceAwareBlockAllocator` interface and provides
    functionality for allocating and managing blocks of memory on both CPU and
    GPU devices.

    The `CpuGpuBlockAllocator` maintains separate memory pools for CPU and GPU
    blocks, and allows for allocation, deallocation, forking, and swapping of
    blocks across these memory pools.
    """

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> DeviceAwareBlockAllocator:
        """Creates a CpuGpuBlockAllocator instance with the specified
        configuration.

        This static method creates and returns a CpuGpuBlockAllocator instance
        based on the provided parameters. It initializes the CPU and GPU block
        allocators with the specified number of blocks, block size, and
        allocator type.

        Args:
            allocator_type (str): The type of block allocator to use for CPU
                and GPU blocks. Currently supported values are "naive" and
                "prefix_caching".
            num_gpu_blocks (int): The number of blocks to allocate for GPU
                memory.
            num_cpu_blocks (int): The number of blocks to allocate for CPU
                memory.
            block_size (int): The size of each block in number of tokens.

        Returns:
            DeviceAwareBlockAllocator: A CpuGpuBlockAllocator instance with the
                specified configuration.

        Notes:
            - The block IDs are assigned contiguously, with GPU block IDs coming
                before CPU block IDs.
        """
        block_ids = list(range(num_gpu_blocks + num_cpu_blocks))
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        if allocator_type == "naive":
            gpu_allocator = NaiveBlockAllocator(
                create_block=NaiveBlock,
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator = NaiveBlockAllocator(
                create_block=NaiveBlock,
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        elif allocator_type == "prefix_caching":
            gpu_allocator = PrefixCachingBlockAllocator(
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator = PrefixCachingBlockAllocator(
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        else:
            raise ValueError(f"Unknown allocator type {allocator_type=}")

        return CpuGpuBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )

    def __init__(self, cpu_block_allocator: BlockAllocator,
                 gpu_block_allocator: BlockAllocator):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        self._allocators = {
            Device.CPU: cpu_block_allocator,
            Device.GPU: gpu_block_allocator,
        }

        self._block_ids_to_allocator = {}
        self._swap_mapping = {}
        for _, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator

    def allocate_mutable(self, prev_block: Optional[Block],
                         device: Device) -> Block:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated mutable block.
        """
        return self._allocators[device].allocate_mutable(prev_block)

    def allocate_immutable(self, prev_block: Optional[Block],
                           token_ids: List[int], device: Device) -> Block:
        """Allocates a new immutable block with the provided token IDs on the
        specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            token_ids (List[int]): The list of token IDs to be stored in the new
                block.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated immutable block containing the provided
                token IDs.
        """
        return self._allocators[device].allocate_immutable(
            prev_block, token_ids)

    def mock_mutable(self, prev_block: Optional[Block], token_ids: List[int],
                     device: Device) -> Block:
        """Mock a new mutable block, linked to the previous block, to help with
        content hash calculation.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.

        Returns:
            Block: The newly allocated mutable block.
        """
        return self._allocators[device].mock_mutable(prev_block, token_ids)

    def reference(self, block_id: int) -> None:
        """Notify the device aware allocator there is new sequence reference
        the given block.

        Args:
            block (Block): The block to be referenced.
        """
        allocator = self._block_ids_to_allocator[block_id]
        return allocator.reference(block_id)

    def free(self, block: Block) -> None:
        """Frees the memory occupied by the given block.

        Args:
            block (Block): The block to be freed.
        """
        allocator = self._block_ids_to_allocator[block.block_id]
        return allocator.free(block)

    def fork(self, last_block: Block) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
            memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: A new list of blocks that shares the same memory as the
                original sequence.
        """
        allocator = self._block_ids_to_allocator[last_block.block_id]
        return allocator.fork(last_block)

    def get_num_free_blocks(self, device: Device) -> int:
        """Returns the number of free blocks available on the specified device.

        Args:
            device (Device): The device for which to query the number of free
                blocks.

        Returns:
            int: The number of free blocks available on the specified device.
        """
        return self._allocators[device].get_num_free_blocks()

    def clear_copy_on_writes(self) -> Dict[int, List[int]]:
        """Clears the copy-on-write (CoW) state and returns the mapping of
            source to destination block IDs.

        Returns:
            Dict[int, List[int]]: A dictionary mapping source block IDs to lists
                of destination block IDs.
        """
        # CoW only supported on GPU
        device = Device.GPU
        return self._allocators[device].clear_copy_on_writes()

    def mark_blocks_as_computed(self) -> None:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].mark_blocks_as_computed()

    def get_common_computed_block_ids(
            self, seq_block_ids: List[List[int]]) -> List[int]:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].get_common_computed_block_ids(
            seq_block_ids)

    def all_block_ids(self) -> frozenset[int]:
        return frozenset(self._block_ids_to_allocator.keys())

    def update_seq_swap_out_block_mapping(self, block: Block,
                                          block_table: BlockTable,
                                          destination_device: Device) -> None:
        if block.block_id in self._swap_mapping:
            dest_block_id = self._swap_mapping[block.block_id]
            self.reference(dest_block_id)
        else:
            dest_block = block_table.allocate(token_ids=block.token_ids,
                                              device=destination_device,
                                              by_block=True)
            self._swap_mapping[block.block_id] = dest_block.block_id

    def get_and_reset_swaps(self) -> dict:
        mapping = self._swap_mapping.copy()
        self._swap_mapping.clear()
        return mapping

    # def get_seq_swap_out_block_mapping(
    #         self, seq: Sequence, block_table: BlockTable,
    #         mapping: Dict[Block, Block]) -> BlockTable:
    #     # The swap out logic for a sequence, the mapping dict will be updated
    #     # and the new block table for swapped out sequence is returned.
    #     new_block_table = BlockTable(
    #         block_size=self._block_size,
    #         block_allocator=self,
    #     )
    #     for src_block in block_table.get_blocks():
    #         if src_block in mapping:
    #             cpu_block = mapping[src_block]
    #             self.reference(cpu_block)
    #         else:
    #             cpu_block = new_block_table.allocate(
    #                 token_ids=src_block.token_ids,
    #                 device=Device.CPU,
    #                 by_block=True)
    #             mapping[src_block] = cpu_block
    #         self.free(src_block)
    #     return new_block_table

    # def get_seq_swap_in_block_mapping(
    #         self, seq: Sequence, block_table: BlockTable,
    #         mapping: Dict[Block, Block]) -> BlockTable:
    #     # The swap in logic for a sequence, the mapping dict will be updated
    #     # and the new block table for swapped in sequence is returned.
    #     new_block_table = BlockTable(
    #         block_size=self._block_size,
    #         block_allocator=self,
    #     )
    #     for cpu_block in block_table.get_blocks():
    #         if cpu_block in mapping:
    #             gpu_block = mapping[cpu_block]
    #             self.reference(gpu_block)
    #         else:
    #             gpu_block = new_block_table.allocate(
    #                 token_ids=cpu_block.token_ids,
    #                 device=Device.GPU,
    #                 by_block=True)
    #             mapping[cpu_block] = gpu_block
    #         self.free(cpu_block)
    #     return new_block_table
