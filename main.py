from argparse import ArgumentParser
from src.data.dataset_builder import MultilingualDataset
from src.funnelling import *
from src.view_generators import *
from src.util.common import get_params

def main(args):
    dataset = MultilingualDataset()
    dataset.load_dataset(args.dataset)

    lX, lY = dataset.training()
    lXte, lYte = dataset.test()

    # Initialize Generalized Funnelling
    embedder_list = []
    posterior_VGF = VanillaFunGen(base_learner=get_learner(calibrate=True), n_jobs=2)

    embedder_list.append(posterior_VGF)
    doc_embedders = DocEmbedderList(embedder_list=embedder_list, probabilistic=True)
    
    meta = MetaClassifier(meta_learner=get_learner(calibrate=False, kernel='rbf'),
                      meta_parameters=get_params(optimc=True),
                      n_jobs=2)
    
    gfun = Funnelling(first_tier=doc_embedders, meta_classifier=meta, n_jobs=2)

    # fitting
    gfun.fit(lX, lY)

    # predict
    preds = gfun.predict(lXte)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Generalized Funnelling")
    parser.add_argument("--dataset", type=str, default="data/selected_multilingual_wikinews.jsonl",
                        help="Path to the dataset file")
    args = parser.parse_args()

    main(args)