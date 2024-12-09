#!/usr/bin/env python3

import os
import pandas as pd
import argparse


def parse_args():
    p = argparse.ArgumentParser('generates dataframe from a links file, to be '
                                'consumed by julius CLI dispatcher')
    p.add_argument('--links-fname', type=os.path.abspath, required=True,
                   help='path to links file enumerating data directories')
    p.add_argument('--df-out-fname', type=os.path.abspath, required=True,
                   help='path to write out dataframe pickle to')
    return p.parse_args()


def main():
    # read args from the command line
    args = parse_args()

    # load up the annotated links file
    links_fname = args.links_fname
    df = pd.read_csv(links_fname, sep=' ')

    # give each paper a canonical identifier - the index is already canonical
    # and unique so let's just use that
    df['uq_id'] = df.index

    # write it out to a pickle file
    pkl_fname = args.df_out_fname
    df.to_pickle(pkl_fname)


if __name__ == "__main__":
    main()
