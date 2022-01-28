import logging

from apps import sfm

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(filename)s:%(lineno)s\t %(message)s",
    )
    sfm.run_sfm()
