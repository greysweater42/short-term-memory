from src.raw_person import write_raw_persons_to_cache
import logging


if __name__ == '__main__':
    # TODO this should be in click, so it had a --help attribute, which would describe what it does
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    write_raw_persons_to_cache()
