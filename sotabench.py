from sotabencheval.utils import is_server, set_env_on_server, SOTABENCH_CACHE
from sotabencheval.question_answering import SQuADEvaluator, SQuADVersion
import torch
from torch.hub import download_url_to_file
from pathlib import Path


set_env_on_server("PYTORCH_PRETRAINED_BERT_CACHE", SOTABENCH_CACHE / "pytorch_pretrained_bert")
import sys
sys.path = ["code"] + sys.path
from code import run_squad


def get_default_settings(version: SQuADVersion):
    class _Args:
        n_best_size = 20
        max_answer_length = 30
        max_seq_length = 384
        doc_stride = 128
        max_query_length = 64
        batch_size = 8
        do_lower_case = False
        verbose_logging = False
        version_2_with_negative = (version == SQuADVersion.V20)
        is_training = False
    return _Args()


def prepare_data(eval_examples, tokenizer, settings):
    eval_features = run_squad.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=settings.max_seq_length,
        doc_stride=settings.doc_stride,
        max_query_length=settings.max_query_length,
        is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = run_squad.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = run_squad.DataLoader(eval_data, batch_size=settings.batch_size)
    return eval_dataloader, eval_features


# todo: find threshold used in the paper
def evaluate(model, tokenizer, device, eval_examples, settings, na_prob_thresh=1.0):
    eval_dataloader, eval_features = prepare_data(eval_examples, tokenizer, settings)

    # no need to provide eval_dataset when pred_only=True
    result, preds, nbest_preds = \
        run_squad.evaluate(settings, model, device, None, eval_dataloader,
                           eval_examples, eval_features, na_prob_thresh=na_prob_thresh,
                           pred_only=True)
    return preds


def run_benchmark(model_url: str, model_name: str, version: SQuADVersion):
    evaluator = SQuADEvaluator(
        local_root="data/nlp/squad",
        model_name=model_name,
        paper_arxiv_id="1907.10529",
        version=version
    )

    model = run_squad.BertForQuestionAnswering.from_pretrained(model_url)
    settings = get_default_settings(evaluator.version)
    tokenizer = run_squad.BertTokenizer.from_pretrained("spanbert-large-cased", do_lower_case=False)

    device = torch.device("cuda")
    model.to(device)

    eval_examples = run_squad.read_squad_examples(
        input_file=evaluator.dataset_path, is_training=False,
        version_2_with_negative=settings.version_2_with_negative)

    # when on sotabench server, run the pipeline on a small dataset first and
    # compare the results with cache to avoid recomputing on whole dataset
    cache_exists = False
    if is_server():
        small_examples = eval_examples[::100]
        answers = evaluate(model, tokenizer, device, small_examples, settings)
        evaluator.add(answers)
        if evaluator.cache_exists:
            cache_exists = True
        else:
            evaluator.reset()

    evaluator.reset_time()
    if not cache_exists or not is_server():
        answers = evaluate(model, tokenizer, device, eval_examples, settings)
        evaluator.add(answers)

    evaluator.save()
    print(evaluator.results)


def get_datasets(versions):
    squad_links = {
        SQuADVersion.V11:"https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        SQuADVersion.V20: "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    }
    filenames = {
        SQuADVersion.V11: "dev-v1.1.json",
        SQuADVersion.V20: "dev-v2.0.json"
    }
    data_dir = Path(".data") if is_server() else Path("data")
    datasets_path = data_dir / "nlp" / "squad"
    datasets_path.mkdir(parents=True, exist_ok=True)
    for version in versions:
        filename = datasets_path / filenames[version]
        if not filename.exists():
            download_url_to_file(squad_links[version], filename)


get_datasets([SQuADVersion.V11, SQuADVersion.V20])

run_benchmark("http://dl.fbaipublicfiles.com/fairseq/models/spanbert_squad1.tar.gz", "SpanBERT", SQuADVersion.V11)
run_benchmark("http://dl.fbaipublicfiles.com/fairseq/models/spanbert_squad2.tar.gz", "SpanBERT", SQuADVersion.V20)
