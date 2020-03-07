import sys
import tempfile
import time

import torch
import tqdm

from decofre import datatools

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('forkserver', force=True)
    num_loaders = int(sys.argv[4])
    spans_file = sys.argv[1]
    words_lex, chars_lex, types_lex = datatools.generate_lexicons(spans_file)
    with tempfile.TemporaryDirectory(prefix="decofre_speedtest_") as temp_dir:
        train_set = datatools.SpansDataset.from_tsv(
            spans_file,
            words_lexicon=words_lex,
            chars_lexicon=chars_lex,
            tags_lexicon=types_lex,
            cache_dir=temp_dir,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=30,
            collate_fn=datatools.SpansDataset.collate,
            shuffle=True,
            num_workers=num_loaders ,
        )
        before = time.perf_counter()
        for batch in tqdm.tqdm(
            train_loader,
            desc="Iterating over span batch",
            unit="batch",
            unit_divisor=1024,
            unit_scale=True,
            dynamic_ncols=True,
            leave=False,
        ):
            pass
        diff = time.perf_counter() - before
        print(f"loading time: {diff/len(train_loader)} s⋅batch⁻¹")

    mentions_file, pairs_file = sys.argv[2:4]
    with tempfile.TemporaryDirectory(prefix="decofre_speedtest") as temp_dir:
        train_set = datatools.AntecedentsDataset.from_tsv(
            mentions_file,
            pairs_file,
            words_lexicon=words_lex,
            chars_lexicon=chars_lex,
            cache_dir=temp_dir,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=10,
            collate_fn=datatools.AntecedentsDataset.collate,
            shuffle=True,
            num_workers=num_loaders ,
        )
        before = time.perf_counter()
        for batch in tqdm.tqdm(
            train_loader,
            desc="Iterating over antecedents batch",
            unit="batch",
            unit_divisor=1024,
            unit_scale=True,
            dynamic_ncols=True,
            leave=False,
        ):
            pass
        diff = time.perf_counter() - before
        print(f"loading time: {diff/len(train_loader)} s⋅batch⁻¹")
