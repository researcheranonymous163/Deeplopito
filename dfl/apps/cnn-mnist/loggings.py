import logging
import pathlib
from datetime import datetime

from pythonjsonlogger import jsonlogger as json_logging

logging.getLogger('qbittorrentapi.request').disabled = True
logging.getLogger('qbittorrentapi.auth').disabled = True
logging.getLogger('urllib3.connectionpool').disabled = True
logging.getLogger('torrentfile.torrent').disabled = True
logging.getLogger('asyncio').disabled = True
logging.getLogger('urllib3.util.retry').disabled = True
logging.getLogger('qt.network.http2').disabled = True


# Monkey-patch the default time formatter to use the ISO-8601 format.
def logging_formatter_format_time(self, record, datefmt=None):
	return datetime.fromtimestamp(record.created).astimezone().isoformat()


logging.Formatter.formatTime = logging_formatter_format_time

# See https://github.com/madzak/python-json-logger/pull/183 and https://github.com/madzak/python-json-logger/issues/187.
json_logging.RESERVED_ATTRS = (*json_logging.RESERVED_ATTRS, 'taskName')

# Add console stderr handler to the root logger.

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

stderr_handler = logging.StreamHandler()
stderr_handler.setFormatter(logging.Formatter(fmt='%(name)s\t%(asctime)s\t%(levelname)s\t%(message)s'))
root_logger.addHandler(stderr_handler)


class CustomJsonFormatter(json_logging.JsonFormatter):
	def add_fields(self, log_record, record, message_dict) -> None:
		super().add_fields(log_record, record, message_dict)
		if 'datetime' not in log_record:
			log_record['datetime'] = datetime.now().astimezone().isoformat()
		if 'level' not in log_record:
			log_record['level'] = record.levelname.upper()
		
		if 'name' not in log_record:
			log_record['name'] = record.name


def get_logger(name: str, log_file_name=None) -> logging.Logger:
	if log_file_name is None:
		log_file_name = f"./logs/{name}.jsonl"
	
	log_file = pathlib.Path(log_file_name)
	
	logger = logging.getLogger(name)
	if len(logger.handlers) == 0:
		if log_file.exists():
			raise RuntimeError(f"The chosen log file already exists; not overwriting, nor appending: {log_file}")
		pathlib.Path(log_file.parent).mkdir(parents=True, exist_ok=True)
		
		log_file_handler = logging.FileHandler(log_file)
		log_file_handler.setFormatter(CustomJsonFormatter())
		log_file_handler.setLevel(logging.INFO)
		logger.addHandler(log_file_handler)
	
	return logger
