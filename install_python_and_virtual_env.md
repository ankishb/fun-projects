# This is the documentation on installing Python (hard way)

## If you prefer to install in easy way
	1. Find ppa for python package (compatible version)
	2. sudo apt-get update
	3. sudo apt-get -y install python3.7


## I prefer to install from source, Following are the steps for that
1. Go to root of your system
	cd ~

2. Copy the link of python package from official website, for exp
	wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz

3. Extract code
	tar -xf Python-3.7.4.tgz

4. Move to Python directory 
	cd python-3.7.4/

5. Configuration of python with system
	./configure --enable-optimizations
	(For further optimization of python binary file, we can use --enable-optimizations, otherwise we can ignore it too) 
	

6. Build all package
	make

7. Install required pakage
	sudo make install

8. Clean extra (unrequired) package
	make clean

9. Check if python path is all set
	which python3.7

10. Check version
	python3.7 --version

## Create virtual environment to separate out package from the root
1. Install Virtualenv
	sudo apt-get update
	sudo pip install virtualenv

2. Check Python package location
	which python3

3. Create virtaul environment
	virtualenv -p /usr/bin/python3.7 scipy

4. To activate virtual environment
	source scipy/bin/activate

5. To deactivate
	source deactivate
