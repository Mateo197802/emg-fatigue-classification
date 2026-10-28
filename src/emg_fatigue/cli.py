from dataclasses import replace
from pathlib import Path
import argparse, json, os
import pandas as pd
from .contracts import read_config, output_root, write_json, sha256_file
from .workflow import inventory, process_dataset, eligible_features, synthetic_recordings, environment, verify_feature_cache
from .io import load_recordings, read_csv_bytes, reassemble_fragments
from .validation import evaluate_nested
from .inference import fit_final, load_bundle, predict_recording


def main():
    parser=argparse.ArgumentParser(description='Offline binary sEMG weak-label classification. No independent fatigue validation is implied.')
    parser.add_argument('command',choices=['inventory','preprocess','labels','features','evaluate','fit-final','predict','demo','sensitivities','report'])
    parser.add_argument('--config',default='configs/study.yaml')
    parser.add_argument('--input',help='CSV for prediction or ZIP/RAR for study processing')
    parser.add_argument('--fragments',nargs='+',help='Contiguous CSV fragments with original increasing timestamps')
    parser.add_argument('--output-dir')
    parser.add_argument('--model')
    args=parser.parse_args();c=read_config(args.config);out=output_root(args.output_dir)
    if args.command=='predict':
        if bool(args.input)==bool(args.fragments):parser.error('prediction requires --input or --fragments')
        rec=reassemble_fragments(args.fragments) if args.fragments else read_csv_bytes(Path(args.input).read_bytes(),'inference','unseen','csv','unknown','unknown')
        bundle=load_bundle(args.model or out/'model.joblib');pred=predict_recording(rec,bundle)
        pred.to_csv(out/'predictions.csv',index=False)
        write_json(out/'prediction_receipt.json',{'input_sha256':rec.sha256,'model_sha256':sha256_file(args.model or out/'model.joblib'),'windows':len(pred),'predicted':int((pred.status=='predicted').sum()),'scope':'offline weak-label model, not physiological validation'})
    elif args.command=='demo':
        data=process_dataset(synthetic_recordings(),c,out)
        f=eligible_features(data['features'])
        evaluate_nested(f,c,out/'evaluation',candidates=[{'family':'majority'},{'family':'logistic_rms_mdf','C':1.}])
        fit_final(f,c,out/'model.joblib',candidates=[{'family':'logistic_rms_mdf','C':1.}])
        write_json(out/'DEMO_ONLY.json',{'synthetic':True,'physiological_simulation':False})
    elif args.command in ['inventory','preprocess','labels','features']:
        source=args.input or os.environ.get('EMG_DATA_ZIP')
        if not source:parser.error('set EMG_DATA_ZIP or --input')
        recordings=load_recordings(source,c)
        if args.command=='inventory':inventory(recordings,out)
        else:process_dataset(recordings,c,out)
    elif args.command in ['evaluate','fit-final']:
        receipt=verify_feature_cache(out,c)
        f=eligible_features(pd.read_parquet(out/'features.parquet'))
        if args.command=='evaluate':evaluate_nested(f,c,out/'evaluation')
        else:fit_final(f,c,out/'model.joblib')
    elif args.command=='sensitivities':
        from .sensitivities import run_sensitivities
        source=args.input or os.environ.get('EMG_DATA_ZIP')
        if not source:parser.error('set EMG_DATA_ZIP or --input')
        run_sensitivities(load_recordings(source,c),c,out)
    elif args.command=='report':
        from .reporting import render_report_assets
        render_report_assets(out,c)
    print(f'{args.command}: completed; private output: {out}',flush=True)


if __name__=='__main__':main()
