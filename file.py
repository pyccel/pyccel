if __name__ == '__main__':
	#$omp loop bind(thread)
	for i in range(3):
		print('Hello World')