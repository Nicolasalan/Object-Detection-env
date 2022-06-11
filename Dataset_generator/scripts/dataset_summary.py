import pandas as pd
import argparse
from tqdm import tqdm
import os

def generate_summary(name, input_csv, output_dir):
   df = pd.read_csv(input_csv)

   files = df['filename'].unique().tolist()
   classes = df['class'].unique().tolist()

   stat_bbox = df.groupby('class')['filename'].size()
   
   with open(os.path.join(output_dir, name + '_report.txt'), 'w', encoding='utf-8') as f:
       f.write("#############  Dataset %s  #############\n\n\n" % name)

       f.write("Número de imagens = %s\n\n\n" % len(files))
       f.write("Número de classes = %s\n\n" % len(classes))
       
       f.write("Classes: %s\n\n\n" % ', '.join(classes))

       f.write("Número de aparições de cada classe (bounding boxes):\n\n")
       for i, v in stat_bbox.items():
           f.write('{0: <20} {1: <10}\n'.format(i, v))

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description=
       'Reads the contents of a CSV file, containing object annotations and their corresponding images\'s and generate a summury report'
   )

   parser.add_argument('name',
                       metavar='name',
                       type=str,
                       help='description name')

   parser.add_argument('input_csv',
                       metavar='input_csv',
                       type=str,
                       help='Path to the input CSV file')

   parser.add_argument(
       'output_dir',
       metavar='output_dir',
       type=str,
       help='Directory where the .txt report file will be generated'
   )

   args = parser.parse_args()

   generate_summary(args.name, args.input_csv, args.output_dir)