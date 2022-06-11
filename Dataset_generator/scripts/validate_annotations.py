import pandas as pd
import argparse
from tqdm import tqdm
import os

def validate_csv(input_csv, image_dir, output_dir):
   df = pd.read_csv(input_csv)

   files = df['filename']

   not_founded = []

   for fl in tqdm(files, desc='files'):
       if not os.path.exists(os.path.join(image_dir,fl)):
           if fl not in not_founded:
               not_founded.append(fl)

   with open(os.path.join(output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
       for item in not_founded:
           f.write("File does not exists: %s\n" % item)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description=
       'Reads the contents of a CSV file, containing object annotations and checks if corresponding images exists'
   )

   parser.add_argument('input_csv',
                       metavar='input_csv',
                       type=str,
                       help='Path to the input CSV file')

   parser.add_argument('image_dir',
                       metavar='image_dir',
                       type=str,
                       help='Path to the directory containing all images')

   parser.add_argument(
       'output_dir',
       metavar='output_dir',
       type=str,
       help='Directory where the .txt report file will be generated'
   )

   args = parser.parse_args()

   validate_csv(args.input_csv, args.image_dir, args.output_dir)
