import sys
import os
import urllib.request
import tarfile
import zipfile


def report_download_progress(count,block_size,total_size):
	percent_comp = float(count*block_size*100)/float(total_size)
	msg = "\r {0:.1%} already downloaded".format(percent_comp)
	sys.stdout.write(msg)
	sys.stdout.flush()


def download_data_url(url,download_dir):
	filename = url.split('/')[-1]
	path = os.path.join(download_dir,filename)

	if not os.path.exists(path):
		os.makedirs(download_dir,exist_ok=True)

		print("Download %s to %s"%(url,path))

		path,_ = urllib.request.urlretrieve(url=url,filename=path,reporthook=report_download_progress)

		print("\nExtracting files")
		if path.endswith(".zip"):
			zipfile.Zipfile(file=path,mode="r").extractall(download_dir)
		elif path.endswith((".tar.gz",".tgz")):
			tarfile.open(name=path,mode="r:gz").extractall(download_dir)