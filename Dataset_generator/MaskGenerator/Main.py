import argparse
from src.MaskGenerator import MaskGenerator

def parseArguments():
    parser = argparse.ArgumentParser(
       description='Generates object masks for all images in given directory and subdirectories. Mask images are saved in the same directory.',
       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image_dir',
                        metavar='image_dir',
                        type=str,
                        help='Path to the directory containing images')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    MaskGenerator().generateForListOfImages(args.image_dir)
