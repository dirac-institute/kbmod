"""Functions for performing masking operations as specified in the configuration.
"""

import kbmod.search as kb


def mask_flags_from_dict(mask_bits_dict, flag_keys):
    """Generate a bitmask integer of flag keys from a dictionary
    of masking reasons and a list of reasons to use.

    Attributes
    ----------
    mask_bits_dict : `dict`
        A dictionary mapping a masking key (string) to the bit
        number in the masking bit vector.
    flag_keys : `list`
        A list of masking keys to use.

    Returns
    -------
    bitmask : `int`
        The bitmask to use for masking operations.
    """
    bitmask = 0
    for bit in flag_keys:
        bitmask += 2 ** mask_bits_dict[bit]
    return bitmask


def apply_mask_operations(config, stack):
    """Perform all the masking operations based on the search's configuration parameters.

    Parameters
    ----------
    config : `SearchConfiguration`
        The configuration parameters
    stack : `ImageStack`
        The stack before the masks have been applied. Modified in-place.

    Returns
    -------
    stack : `ImageStack`
        The stack after the masks have been applied.
    """
    # Generate the global mask before we start modifying the individual masks.
    if config["repeated_flag_keys"] and len(config["repeated_flag_keys"]) > 0:
        global_flags = mask_flags_from_dict(config["mask_bits_dict"], config["repeated_flag_keys"])
        global_binary_mask = stack.make_global_mask(global_flags, config["mask_num_images"])
    else:
        global_binary_mask = None

    # Start by creating a binary mask out of the primary flag values. Prioritize
    # the config's mask_bit_vector over the dictionary based version.
    if config["mask_bit_vector"]:
        mask_flags = config["mask_bit_vector"]
    elif config["flag_keys"] and len(config["flag_keys"]) > 0:
        mask_flags = mask_flags_from_dict(config["mask_bits_dict"], config["flag_keys"])
    else:
        mask_flags = 0

    # Apply the primary mask.
    for i in range(stack.img_count()):
        stack.get_single_image(i).binarize_mask(mask_flags)

    # If the threshold is set, mask those pixels.
    if config["mask_threshold"]:
        for i in range(stack.img_count()):
            stack.get_single_image(i).union_threshold_masking(config["mask_threshold"])

    # Union in the global masking if there was one.
    if global_binary_mask is not None:
        for i in range(stack.img_count()):
            stack.get_single_image(i).union_masks(global_binary_mask)

    # Grow the masks.
    if config["mask_grow"] and config["mask_grow"] > 0:
        for i in range(stack.img_count()):
            stack.get_single_image(i).grow_mask(config["mask_grow"])

    # Apply the masks to the images.
    for i in range(stack.img_count()):
        stack.get_single_image(i).apply_mask(0xFFFFFF)

    return stack
