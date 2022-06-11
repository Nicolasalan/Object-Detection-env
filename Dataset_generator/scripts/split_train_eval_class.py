import os
import argparse
import pandas as pd
import numpy as np

def some_row(df):
    return df.loc[np.random.choice(df.index, size=1)].iloc[0].tolist()

def train_test_split(df, train_size, txt_path):
    size = len(df['filename'].tolist())
    num_bbox_train = int(train_size*size)
    
    train_df = df.copy()
    eval_df = pd.DataFrame(columns = list(train_df.columns.values))


    class_list = [
        l.rstrip('\n').strip()
        for l in open(txt_path, 'r', encoding='utf-8-sig')
    ]

    class_list = [l for l in class_list if len(l) > 0]
   
    for cl in class_list:
        row = some_row(train_df[train_df['class']==cl])

        eval_df = eval_df.append(train_df[train_df['filename']==row[0]], ignore_index=True)
        train_df = train_df[train_df['filename']!=row[0]]

    while len(train_df['filename'].tolist()) > num_bbox_train:
        stat_bbox = eval_df.groupby('class')['filename'].size()
        bbox_list = [(i, v) for i, v in stat_bbox.items()]
        sorted_bbox_list = sorted(bbox_list, key=lambda tup: tup[1])
        
        i=0
        while True:
            cl, _ = sorted_bbox_list[i]
            if cl in train_df['class'].unique().tolist():
                break
            i += 1

        row = some_row(train_df[train_df['class']==cl])

        eval_df = eval_df.append(train_df[train_df['filename']==row[0]], ignore_index=True)

        train_df = train_df[train_df['filename']!=row[0]]

    return train_df, eval_df

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description='Separates a CSV file into training and validation sets',
       formatter_class=argparse.RawDescriptionHelpFormatter)
   parser.add_argument('input_csv',
                       metavar='input_csv',
                       type=str,
                       help='Path to the input CSV file')
   parser.add_argument('class_list',
                       metavar='class_list',
                       type=str,
                       help='Path to the list of classes. A txt file with one class name by line')
   parser.add_argument(
       '-f',
       metavar='train_frac',
       type=float,
       default=.80,
       help='fraction of the dataset that will be separated for training (default .75)')
   parser.add_argument(
       '-o',
       metavar='output_dir',
       type=str,
       default=None,
       help='Directory to output train and evaluation datasets (default input_csv directory)')

   args = parser.parse_args()

   if args.f < 0 or args.f > 1:
      raise ValueError('train_frac must be between 0 and 1')

   # output_dir = input_csv directory is None
   if args.o is None:
      output_dir, _ = os.path.split(args.input_csv)
   else:
      output_dir = args.o

   df = pd.read_csv(args.input_csv)

   train_df, validation_df = train_test_split(df, train_size=args.f, txt_path=args.class_list)

   # output files have the same name of the input file, with some extra stuff appended
   new_csv_name = os.path.splitext(args.input_csv)[0]
   train_csv_path = os.path.join(output_dir, new_csv_name + '_train.csv')
   eval_csv_path = os.path.join(output_dir, new_csv_name + '_eval.csv')

   train_df.to_csv(train_csv_path, index=False)
   validation_df.to_csv(eval_csv_path, index=False)
