#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cloudpickle
import argparse

# stitch the lib python into the path so julius imports will work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib-python'))
import julius.models
import julius.records
import julius.fits
import julius.models
import julius.visualization


DEFAULT_RECORDS_DIR = os.path.abspath('/Users/aditya/research/017_bayes_tafel_fitting/litsearch/batch1/figure-images')
DEFAULT_DF_NAME = os.path.abspath('/Users/aditya/research/017_bayes_tafel_fitting/litsearch/batch1/batch1-dataframe.pkl')


# utility functions for dumping and loading pickles, need cloudpickle because
# we will be pickling function objects pretty regularly
def dump_pickle(obj, fname):
    with open(fname, 'wb') as f:
        cloudpickle.dump(obj, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        obj = cloudpickle.load(f)

    return obj


def maybe_make_directory(d):
    if not os.path.exists(d):
        os.mkdir(d)


def process_one_directory(root, uq_idx):
    '''
        Function to go through a ripped data directory and construct a
        TafelRecord from the data stored in the data and metadata files stored
        in the directory, using a convention I just came up with. a client
        should override this function to produce a TafelRecord from their data,
        which can then be fed to the rest of the julius machinery
    '''
    # search for files like dat_XXX.txt and the corresponding metadata_XXX.txt
    glob_expr = os.path.join(root, 'dat_*.txt')
    glob_search = glob.glob(glob_expr)
    parsed_records = []
    for match in glob_search:
        rid = julius.records.parse_record_id(match)
        mdata_file = os.path.join(root, 'metadata_%03d.txt' % rid)
        data_file = match
        identifier = '%03d_%03d' % (uq_idx, rid)
        tr = julius.records.TafelRecord(data_file, mdata_file, identifier)
        parsed_records.append(tr)

    return parsed_records


def manually_examine_records(records):
    for r in records:
        plt.figure()
        plt.plot(r.s_voltage, r.s_current, 'ko')
        plt.xlabel('$V_{\mathrm{appl.}}$ [V]')
        plt.ylabel(r'$\log i_{\mathrm{prod.}}$')
        plt.show()


def dispatch_single_bayes_series_resistances_fit(record, save_directory,
                                                 nsamples=1000, verbose=True):
    # pull out the current and voltage data, fit a model
    voltages = record.s_voltage
    currents = record.s_current

    # the model has issues when all the y data is negative; for now just
    # shift all the data so it is uniformly positive
    currents = currents - np.min(currents)

    # set the relative error on experimental measurements
    sigma = 0.10

    # no frozen parameters in this model
    frozen_params = None

    # set up the simple model and the probabilistic model for series
    # resistances, as well as an initial guess that was manually tuned,
    # helps to converge fits quickly on the first guess
    simple_model = julius.models.series_resistances_model
    bayes_model = julius.models.series_resistances_model_bayes
    pname_map = julius.models.series_resistances_model_bayes.name_to_index_map
    guess = (15, 2, 15)

    # dispatch the fit
    trace = julius.fits.fit_bayes_with_preconditioned_bounds(voltages,
                                                             currents,
                                                             simple_model,
                                                             bayes_model,
                                                             sigma,
                                                             frozen_params,
                                                             nsamples=nsamples,
                                                             guess=guess)

    # construct the mean a posteriori model from the Bayes traces
    mean_ap_model = \
        julius.fits.collapsed_simple_model_from_bayes_model(trace,
                                                            simple_model,
                                                            frozen_params,
                                                            pname_map)

    # pickle the tafel record, trace, and model in a directory
    pkl_fname = os.path.join(save_directory, '%s.pkl' % record.identifier)
    pickle_me = dict(trace=trace, model=mean_ap_model, record=record)
    dump_pickle(pickle_me, pkl_fname)


def trace_single_bayes_series_resistances_fit(record, nsamples=1000,
                                              verbose=True):
    # pull out the current and voltage data, fit a model
    voltages = record.s_voltage
    currents = record.s_current

    # the model has issues when all the y data is negative; for now just
    # shift all the data so it is uniformly positive
    currents = currents - np.min(currents)

    # set the relative error on experimental measurements
    sigma = 0.10

    # no frozen parameters in this model
    frozen_params = None

    # set up the simple model and the probabilistic model for series
    # resistances, as well as an initial guess that was manually tuned,
    # helps to converge fits quickly on the first guess
    simple_model = julius.models.series_resistances_model
    bayes_model = julius.models.series_resistances_model_bayes
    pname_map = julius.models.series_resistances_model_bayes.name_to_index_map
    guess = (15, 2, 15)

    # dispatch the fit
    trace = julius.fits.fit_bayes_with_preconditioned_bounds(voltages,
                                                             currents,
                                                             simple_model,
                                                             bayes_model,
                                                             sigma,
                                                             frozen_params,
                                                             nsamples=nsamples,
                                                             guess=guess)

    # construct the mean a posteriori model from the Bayes traces
    mean_ap_model = \
        julius.fits.collapsed_simple_model_from_bayes_model(trace,
                                                            simple_model,
                                                            frozen_params,
                                                            pname_map)

    # make the three panel plot
    julius.visualization.three_panel_fit_examination_plot(voltages,
                                                          currents, trace,
                                                          mean_ap_model,
                                                          reported_tafel=record.r_tafel,
                                                          cutoff_fits=None,
                                                          fname=None,
                                                          interactive=True)


def dispatch_bayes_series_resistances_fit(records, save_directory,
                                          nsamples=1000, verbose=True):
    # get the number of fits so we can put down a progress bar
    nfits = len(records)

    # fit all the records!
    for fit_idx, r in enumerate(records):
        # pull out the current and voltage data, fit a model
        voltages = r.s_voltage
        currents = r.s_current

        # the model has issues when all the y data is negative; for now just
        # shift all the data so it is uniformly positive
        currents = currents - np.min(currents)

        # set the relative error on experimental measurements
        sigma = 0.10

        # no frozen parameters in this model
        frozen_params = None

        # set up the simple model and the probabilistic model for series
        # resistances, as well as an initial guess that was manually tuned,
        # helps to converge fits quickly on the first guess
        simple_model = julius.models.series_resistances_model
        bayes_model = julius.models.series_resistances_model_bayes
        pname_map = julius.models.series_resistances_model_bayes.name_to_index_map
        guess = (15, 2, 15)

        # dispatch the fit
        trace = julius.fits.fit_bayes_with_preconditioned_bounds(voltages,
                                                                 currents,
                                                                 simple_model,
                                                                 bayes_model,
                                                                 sigma,
                                                                 frozen_params,
                                                                 nsamples=nsamples,
                                                                 guess=guess)

        # if we trap an error then just move on
        if trace is None:
            print('trapped error on fit index %d' % fit_idx)
            continue

        # construct the mean a posteriori model from the Bayes traces
        mean_ap_model = \
            julius.fits.collapsed_simple_model_from_bayes_model(trace,
                                                                simple_model,
                                                                frozen_params,
                                                                pname_map)

        # pickle the tafel record, trace, and model in a directory
        pkl_fname = os.path.join(save_directory, '%s.pkl' % r.identifier)
        pickle_me = dict(trace=trace, model=mean_ap_model, record=r)
        dump_pickle(pickle_me, pkl_fname)

        if verbose:
            print('finished %d of %d fits' % (fit_idx + 1, nfits))


def visualize_bayes_fits(records, save_directory, fig_directory,
                         interactive=False, no_cutoff=False):
    # read records from the save directory, first glob them all
    glob_expr = os.path.join(save_directory, '*.pkl')
    glob_search = glob.glob(glob_expr)
    nplots = len(glob_search)

    for idx, match in enumerate(glob_search):
        # unpickle the bayes fit and extract its packed components
        p = load_pickle(match)
        r = p['record']
        trace = p['trace']
        model = p['model']

        # pull out the current and voltage data, fit a model
        voltages = r.s_voltage
        currents = r.s_current

        # the model has issues when all the y data is negative; for now just
        # shift all the data so it is uniformly positive
        currents = currents - np.min(currents)

        # if we have enough data points, plot "cutoff fits" which include the
        # first point and a variable number of N points after that point, for
        # various N
        cutoff_fits = None
        enough_voltage_pts = (voltages.size > 4)
        if enough_voltage_pts:
            rightmost_point = 5
            cutoff_fits = \
                julius.fits.fit_several_windowed_linear_models(voltages,
                                                               currents,
                                                               rightmost_point)

        if no_cutoff:
            cutoff_fits = None

        # name for the figure
        fig_fname = os.path.join(fig_directory, '%s.pdf' % r.identifier)

        # make the three panel plot
        julius.visualization.three_panel_fit_examination_plot(voltages,
                                                              currents, trace,
                                                              model,
                                                              reported_tafel=r.r_tafel,
                                                              cutoff_fits=cutoff_fits,
                                                              fname=fig_fname,
                                                              interactive=interactive)

        # print progress message
        print('completed %d of %d plots' % (idx + 1, nplots), end='\r')


def extract_records(records_dir, df):
    # make a list to hold all the parsed Tafel records
    all_records = []

    # go through the directories and parse out all the records, appending to
    # the list as we go
    for row in df.iterrows():
        entry = row[1]
        uqid = entry.uq_id
        this_dir = os.path.join(records_dir, '%03d' % uqid)
        records = process_one_directory(this_dir, uqid)

        all_records.extend(records)

    return all_records


def parse_args():

    p = argparse.ArgumentParser('driver script for Tafel fitting')
    sp = p.add_subparsers(help='commands')

    erec = sp.add_parser('examine-records', help='examine a experimental '
                         'Tafel records')
    erec.set_defaults(which='examine-records')
    erec.add_argument('--records-dir', type=os.path.abspath, required=False,
                      default=DEFAULT_RECORDS_DIR)
    erec.add_argument('--df-name', type=os.path.abspath, required=False,
                      default=DEFAULT_DF_NAME)
    erec.add_argument('--interactive', action='store_true', default=False)

    dfits = sp.add_parser('dispatch-fits', help='dispatch fits')
    dfits.set_defaults(which='dispatch-fits')
    dfits.add_argument('--records-dir', type=os.path.abspath, required=False,
                       default=DEFAULT_RECORDS_DIR)
    dfits.add_argument('--df-name', type=os.path.abspath, required=False,
                       default=DEFAULT_DF_NAME)
    dfits.add_argument('--save-dir', type=os.path.abspath, required=True)
    dfits.add_argument('--rand-seed', type=int, default=2020)
    dfits.add_argument('--nsamples', type=int, default=1000)

    dsfits = sp.add_parser('dispatch-single-fit', help='dispatch a single fit')
    dsfits.set_defaults(which='dispatch-single-fit')
    dsfits.add_argument('--identifier', type=str, required=True, metavar='identifier')
    dsfits.add_argument('--records-dir', type=os.path.abspath, required=False,
                        default=DEFAULT_RECORDS_DIR)
    dsfits.add_argument('--df-name', type=os.path.abspath, required=False,
                        default=DEFAULT_DF_NAME)
    dsfits.add_argument('--save-dir', type=os.path.abspath, required=True)
    dsfits.add_argument('--rand-seed', type=int, default=2020)
    dsfits.add_argument('--nsamples', type=int, default=1000)

    tfits = sp.add_parser('trace-fit', help='trace a single problematic fit')
    tfits.set_defaults(which='trace-fit')
    tfits.add_argument('--record-num', type=int, required=True)
    tfits.add_argument('--records-dir', type=os.path.abspath, required=False,
                       default=DEFAULT_RECORDS_DIR)
    tfits.add_argument('--df-name', type=os.path.abspath, required=False,
                       default=DEFAULT_DF_NAME)
    tfits.add_argument('--rand-seed', type=int, default=2020)
    tfits.add_argument('--nsamples', type=int, default=1000)

    pfits = sp.add_parser('plot-fits', help='make summary plots for fits')
    pfits.set_defaults(which='plot-fits')
    pfits.add_argument('--records-dir', type=os.path.abspath, required=False,
                       default=DEFAULT_RECORDS_DIR)
    pfits.add_argument('--df-name', type=os.path.abspath, required=False,
                       default=DEFAULT_DF_NAME)
    pfits.add_argument('--fits-dir', type=os.path.abspath, required=True)
    pfits.add_argument('--figures-dir', type=os.path.abspath, required=True)
    pfits.add_argument('--interactive', action='store_true', default=False)
    pfits.add_argument('--no-cutoff', action='store_true', default=False)

    return p.parse_args()


def main():
    args = parse_args()

    if args.which == 'dispatch-fits':
        # boot up the dataframe with the paper records and their associated
        # figure directories
        df = pd.read_pickle(args.df_name)
        records = extract_records(args.records_dir, df)

        # seed the random number generator for reproducibility, make the save
        # directory if it doesn't exist, then dispatch the fits
        np.random.seed(args.rand_seed)
        maybe_make_directory(args.save_dir)
        dispatch_bayes_series_resistances_fit(records, args.save_dir,
                                              nsamples=args.nsamples)

    elif args.which == 'trace-fit':
        # boot up the dataframe with the paper records and their associated
        # figure directories
        df = pd.read_pickle(args.df_name)
        records = extract_records(args.records_dir, df)

        # index just the record we want
        record = records[args.record_num]

        # dispatch the traced fit
        np.random.seed(args.rand_seed)
        trace_single_bayes_series_resistances_fit(record,
                                                  nsamples=args.nsamples)

    elif args.which == 'dispatch-single-fit':
        # boot up the dataframe with the paper records and their associated
        # figure directories
        df = pd.read_pickle(args.df_name)
        records = extract_records(args.records_dir, df)

        # index just the record we want
        found = False
        for r in records:
            if r.identifier == args.identifier:
                record = r
                found = True
                break

        if not found:
            print('failed to find record with identifier %s' % args.identifier)
            sys.exit(1)

        # dispatch the traced fit
        np.random.seed(args.rand_seed)
        maybe_make_directory(args.save_dir)
        print('dispatching index %s fit' % args.identifier)
        dispatch_single_bayes_series_resistances_fit(record, args.save_dir,
                                                     nsamples=args.nsamples)

    elif args.which == 'plot-fits':
        # boot up the dataframe with the paper records and their associated
        # figure directories
        df = pd.read_pickle(args.df_name)
        records = extract_records(args.records_dir, df)

        # make the directory for the fits and then do all plots
        maybe_make_directory(args.figures_dir)
        visualize_bayes_fits(records, args.fits_dir, args.figures_dir,
                             interactive=args.interactive,
                             no_cutoff=args.no_cutoff)

    elif args.which == 'examine-record':
        # boot up the dataframe with the paper records and their associated
        # figure directories
        df = pd.read_pickle(args.df_name)
        records = extract_records(args.records_dir, df)
        manually_examine_records(records)


if __name__ == "__main__":
    main()
