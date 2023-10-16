import timeit
import numpy as np

from kbmod.masking import (
    BitVectorMasker,
    DictionaryMasker,
    GlobalDictionaryMasker,
    GrowMask,
    ThresholdMask,
    apply_mask_operations,
)
from kbmod.result_list import ResultRow
from kbmod.search import ImageStack, PSF, RawImage, LayeredImage


def set_up_image_stack():
    """Create a fake image stack to use with the masking tests.

    Returns
    -------
    stack : `ImageStack`
        The generated images.
    """
    imgs = []
    for i in range(10):
        img = LayeredImage(
            "test_image",
            1024,  # Width
            1024,  # Height
            0.1,  # Noise level
            0.01,  # Variance
            i,  # Time
            PSF(1.0),
        )

        # Mask some pixels.
        msk = img.get_mask()
        for y in range(1024):
            for x in range(1024):
                if x % 10 == 0 and y % 10 == 0:
                    msk.set_pixel(x, y, 1)

        imgs.append(img)
    return ImageStack(imgs)


def bench_image_masker(masker, stack):
    """Benchmark a single ImageMasker object.

    Parameters
    ----------
    masker : `ImageMasker`
        The object to benchmark.
    stack : `ImageStack`
        The data to use.

    Returns
    -------
    res_time : `float`
        The average run time.
    """
    tmr = timeit.Timer(stmt="masker.apply_mask(stack)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


def bench_dictionary_masker():
    mask_bits_dict = {
        "BAD": 0,
        "SAT": 1,
        "INTRP": 2,
        "CR": 3,
        "EDGE": 4,
    }
    masker = DictionaryMasker(mask_bits_dict, ["BAD", "EDGE"])
    return bench_image_masker(masker, set_up_image_stack())


def bench_threshold_masker():
    masker = ThresholdMask(100.0)
    return bench_image_masker(masker, set_up_image_stack())


def bench_bit_vector_masker():
    masker = BitVectorMasker(1, [])
    return bench_image_masker(masker, set_up_image_stack())


def bench_cpp_bit_vector():
    stack = set_up_image_stack()
    tmr = timeit.Timer(stmt="stack.apply_mask_flags(1, [])", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


def bench_cpp_threshold():
    stack = set_up_image_stack()
    tmr = timeit.Timer(stmt="stack.apply_mask_threshold(100.0)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


def bench_grow_mask(r):
    res_times = []

    # We run the loop outside because we need to reset the stack each time
    # and we do not want that to be included in the cost.
    for i in range(10):
        stack = set_up_image_stack()
        premasker = BitVectorMasker(1, [])
        premasker.apply_mask(stack)

        grow_masker = GrowMask(r)
        res_times.append(timeit.timeit("grow_masker.apply_mask(stack)", number=1, globals=locals()))
    return np.mean(np.array(res_times))


def bench_grow_mask_cpp(r):
    res_times = []

    # We run the loop outside because we need to reset the stack each time
    # and we do not want that to be included in the cost.
    for i in range(10):
        stack = set_up_image_stack()
        premasker = BitVectorMasker(1, [])
        premasker.apply_mask(stack)

        res_times.append(timeit.timeit("stack.grow_mask(r)", number=1, globals=locals()))
    return np.mean(np.array(res_times))


def run_all_benchmarks():
    print("Apply Mask Timings:")
    print("    Method       |    Time")
    print("-" * 30)
    print(f" Python Dict     | {bench_dictionary_masker():10.7f}")
    print(f" Python Bit Vect | {bench_bit_vector_masker():10.7f}")
    print(f" C++ Bit Vect    | {bench_cpp_bit_vector():10.7f}")
    print(f" Python Thresh   | {bench_threshold_masker():10.7f}")
    print(f" C++ Thresh      | {bench_cpp_threshold():10.7f}")

    print("\n\nGrow Mask Timings:")
    print("r |   Python   |  C++")
    print("-" * 30)
    for r in range(1, 5):
        print(f"{r} | {bench_grow_mask(r):10.7f} | {bench_grow_mask_cpp(r):10.7f}")


if __name__ == "__main__":
    run_all_benchmarks()
