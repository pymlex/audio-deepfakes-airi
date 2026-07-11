"""Task 2.3: verify DatasetWav loading."""

from torch.utils.data import DataLoader

from config import DATA_ROOT, METADATA_DIR
from data.dataset import DatasetWav
from utils.data import make_cm_df


def main() -> None:
    df_train, _, _ = make_cm_df(METADATA_DIR, DATA_ROOT)
    ds = DatasetWav(df_train.head(8))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    print(x.shape, y.shape)


if __name__ == "__main__":
    main()
