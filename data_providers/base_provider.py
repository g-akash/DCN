import numpy as np


class Dataset:
	@property
	def num_examples(self):
		raise NotImplementedError

	def next_batch(self,batch_size):
		raise NotImplementedError



class ImagesDataset(Dataset):

	def _measure_mean_and_std(self):
		means = []
		std = []
		for ch in range(self.images.shape[-1]):
			means.append(np.mean(self.images[:,:,:,ch]))
			stds.append(np.std(self.imegas[:,:,:,ch]))

		self.means = means
		self.std = std


	@property
	def images_means(self):
		if not hasattr(self,'means'):
			self._measure_mean_and_std()

		return self.means

	@property
	def images_stds(self):
		if not hasattr(self,'std'):
			self._measure_mean_and_std()

		return self.std


	def shuffle_images_and_labels(self,images,labels):
		new_index = np.random.permutation(images.shape[0])
		new_images = [images[i] for i in new_index]
		new_labels = [labels[i] for i in new_index]
		return new_images, new_labels

	def normalize_images(self,images,normalization_type):
		if normalization_type=="divide_255":
			images = images/255
		elif normalization_type=="divide_256":
			images = images/256
		elif normalization_type=="by_chanels":
			for i in range(images.shape[-1]):
				images[:,:,:,i]=((images[:,:,:,i]-self.images_means[i])/self.images_stds[i])
		else:
			raise Exception("Unknown type of normalization")
		return images

	def normalize_all_images_by_channels(self,initial_images):
		new_images = np.zeros(initial_images.shape)
		for i in range(initial_images.shape[0]):
			new_images[i] = self.normalize_image_by_channel(initial_images[i])

		return new_images


	def normalize_image_by_channel(self,image):
		new_image = np.zeros(image.shape)
		for ch in range(3):
			mean = np.mean(image[:,:,ch])
			std = np.std(image[:,:,ch])
			new_image[:,:,ch] = (image[:,:,ch]-mean)/std
		return new_image


class DataProvider:
	@property
	def data_shape(self):
		raise NotImplementedError

	@property
	def n_classes(self):
		raise NotImplementedError

	def labels_to_one_hot(self,labels):
		new_labels = np.zeros((labels.shape[0],self.n_classes))
		for i in range(len(labels)):
			new_labels[i][labels[i]]=1.0
		return new_labels

	def labels_from_one_hot(self,labels):
		new_labels = np.argmax(labels,axis=1)
		return new_labels