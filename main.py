import pandas as pd

from argparse import ArgumentParser
from src.data.dataset_builder import MultilingualDataset
from src.funnelling import *
from src.view_generators import *
from src.util.common import get_params


def main(args):
    dataset = MultilingualDataset()
    dataset.load_from_jsonl(args.train_path)

    dataset_predict = MultilingualDataset()
    dataset_predict.load_from_jsonl(args.test_path, is_prediction=True)

    lX, lY = dataset.get_whole_dataset()
    lXte, _ = dataset_predict.get_whole_dataset()

    print("- Training dataset info:")
    for lang, docs in lX.items():
        print(f"lang: {lang}, num_docs: {len(docs)}")
    print("-"* 25)

    print("- Prediction dataset info:")
    for lang, docs in lX.items():
        print(f"lang: {lang}, num_docs: {len(docs)}")
    print("-"* 25)

    # Initialize Generalized Funnelling
    embedder_list = []
    posterior_VGF = VanillaFunGen(base_learner=get_learner(calibrate=True), n_jobs=2)

    embedder_list.append(posterior_VGF)
    doc_embedders = DocEmbedderList(embedder_list=embedder_list, probabilistic=True)
    

    meta = MetaClassifier(meta_learner=get_learner(calibrate=False, kernel='rbf'),
                      meta_parameters=get_params(optimc=True if args.optimc == 1 else False),
                      n_jobs=2)
    
    gfun = Funnelling(first_tier=doc_embedders, meta_classifier=meta, n_jobs=2)

    # fitting
    gfun.fit(lX, lY)

    # predict
    preds = gfun.predict(lXte)

    # store predictions
    test_df = pd.read_json(args.test_path, lines=True) 
    final_dfs = []
    for lang in dataset_predict.langs():
        lang_preds = dataset.mlb.inverse_transform(preds[lang])
        lang_df = test_df[test_df['lang'] == lang].copy()
        lang_df['predicted_label'] = lang_preds
        final_dfs.append(lang_df)
    final_df = pd.concat(final_dfs, ignore_index=True)

    final_df.to_json("predictions.jsonl", orient='records', lines=True) 
    print("Predictions saved to predictions.jsonl")

    exit(0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Generalized Funnelling")
    parser.add_argument("--train_path", type=str, default="data/selected_multilingual_wikinews.jsonl",
                        help="Path to the jsonl dataset file storing documents for the training phase")
    parser.add_argument("--test_path", type=str, default="data/selected_multilingual_wikinews_test.jsonl",
                        help="Path to the jsonl dataset file storing documents to be classified")
    parser.add_argument("--optimc", type=int, default=0,
                        help="Optimize the meta classifier's hyperparameters")
    
    args = parser.parse_args()

    main(args)