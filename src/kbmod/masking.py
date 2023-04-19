"""Classes for performing masking on images from FITS files.

ImageMasker provides an abstract base class that can be overridden to define masking
algorithms for specific studies, instruments, or FITS headers. Specific masking classes
are provided to support common masking operations including: masking based on a bit vector,
masking based on a dictionary, masking based on a threshold, and growing a current mask.
"""

import abc

import kbmod.search as kb


def apply_mask_operations(stack, mask_list):
    """Apply a series of masking operations defined by a list of
    ImageMasker objects.

    Parameters
    ----------
    stack : `kbmod.image_stack`
        The stack before the masks have been applied.
    mask_list : `list`
        A list of mask_list objects.

    Returns
    -------
    stack : `kbmod.image_stack`
        The same stack object to allow chaining.
    """
    for mask in mask_list:
        stack = mask.apply_mask(stack)
    return stack


class ImageMasker(abc.ABC):
    """The base class for masking operations."""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def apply_mask(self, stack):
        """Apply the mask to an image stack.

        Parameters
        ----------
        stack : `kbmod.image_stack`
            The stack before the masks have been applied.

        Returns
        -------
        stack : `kbmod.image_stack`
            The same stack object to allow chaining.
        """
        pass


class BitVectorMasker(ImageMasker):
    """Apply a mask given a bit vector of masking flags to use
    and vector of bit vectors to ignore.

    Attributes
    ----------
    flags : `int`
        A bit vector of masking flags to apply.
    exception_list : `list`
        A list of bit vectors to skip during flagging.
    """

    def __init__(self, flags, exception_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = flags
        self.exception_list = exception_list

    def apply_mask(self, stack):
        """Apply the mask to an image stack.

        Parameters
        ----------
        stack : `kbmod.image_stack`
            The stack before the masks have been applied.

        Returns
        -------
        stack : `kbmod.image_stack`
            The same stack object to allow chaining.
        """
        if self.flags != 0:
            stack.apply_mask_flags(self.flags, self.exception_list)
        return stack


class DictionaryMasker(BitVectorMasker):
    """Apply a mask given a dictionary of masking condition to key
    and a list of masking conditions to use.

    Attributes
    ----------
    mask_bits_dict : `dict`
        A dictionary mapping a masking key (string) to the bit
        number in the masking bit vector.
    flag_keys : `list`
        A list of masking keys to use.
    """

    def __init__(self, mask_bits_dict, flag_keys, *args, **kwargs):
        self.mask_bits_dict = mask_bits_dict
        self.flag_keys = flag_keys

        # Convert the dictionary into a bit vector.
        bitvector = 0
        for bit in self.flag_keys:
            bitvector += 2 ** self.mask_bits_dict[bit]

        # Initialize the BitVectorMasker parameters.
        super().__init__(bitvector, [0], *args, **kwargs)


class GlobalDictionaryMasker(ImageMasker):
    """Apply a mask given a dictionary of masking condition to key
    and a list of masking conditions to use. Masks pixels in every image
    if they are masked in *multiple* images in the stack.

    Attributes
    ----------
    mask_bits_dict : `dict`
        A dictionary mapping a masking key (string) to the bit
        number in the masking bit vector.
    global_flag_keys : `list`
        A list of masking keys to use.
    mask_num_images : `int`
        The number of images that need to be masked in the stack
        to apply the mask to all images.
    """

    def __init__(self, mask_bits_dict, global_flag_keys, mask_num_images, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask_bits_dict = mask_bits_dict
        self.global_flag_keys = global_flag_keys
        self.mask_num_images = mask_num_images

        # Convert the dictionary into a bit vector.
        self.global_flags = 0
        for bit in self.global_flag_keys:
            self.global_flags += 2 ** self.mask_bits_dict[bit]

    def apply_mask(self, stack):
        """Apply the mask to an image stack.

        Parameters
        ----------
        stack : `kbmod.image_stack`
            The stack before the masks have been applied.

        Returns
        -------
        stack : `kbmod.image_stack`
            The same stack object to allow chaining.
        """
        if self.global_flags != 0:
            stack.apply_global_mask(self.global_flags, self.mask_num_images)
        return stack


class ThresholdMask(ImageMasker):
    """Mask pixels over a given value.

    Attributes
    ----------
    mask_threshold : `float`
        The flux threshold for a pixel.
    """

    def __init__(self, mask_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_threshold = mask_threshold

    def apply_mask(self, stack):
        """Apply the mask to an image stack.

        Parameters
        ----------
        stack : `kbmod.image_stack`
            The stack before the masks have been applied.

        Returns
        -------
        stack : `kbmod.image_stack`
            The same stack object to allow chaining.
        """
        stack.apply_mask_threshold(self.mask_threshold)
        return stack


class GrowMask(ImageMasker):
    """Apply a mask that grows the current max out a given number of pixels.

    Attributes
    ----------
    num_pixels : `int`
        The number of pixels to extend the mask.
    """

    def __init__(self, num_pixels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if num_pixels <= 0:
            raise ValueError(f"Invalid num_pixels={num_pixels} for GrowMask")
        self.num_pixels = num_pixels

    def apply_mask(self, stack):
        """Apply the mask to an image stack.

        Parameters
        ----------
        stack : `kbmod.image_stack`
            The stack before the masks have been applied.

        Returns
        -------
        stack : `kbmod.image_stack`
            The same stack object to allow chaining.
        """
        stack.grow_mask(self.num_pixels, True)
        return stack
