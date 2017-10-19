# Python jupyter workbooks exploring doing neuroscience shuffle analyses in the cloud

# Notes on deploying a jupyter container to Google cloud.
I created a jupyter Docker container with pre-installed (nelpy)[https://github.com/nelpy/nelpy] and a different default password by forking the (docker-stacks)[https://github.com/jupyter/docker-stacks] repostiory. My fork is not yet pushed but should be soon.

## Option 1 - Start up a virtual machine with Docker and then manually start container

1. Create an instance using either the web console interface or the command line. Here's an example command line:
```
gcloud compute instances create [instance-name] --image=cos-stable-61-9765-79-0 --image-project=cos-cloud \
   --machine-type=n1-standard-64 
```
This will create a n1-standard-64 instance using the container-optimized OS. Note that ``cos-stable-61-9765-79-0`` was the most recent stable image as of October 2017, but may change over time. Also note I had to be logged in to see the actual name. There is a magic option, ``--metadata-from-file user-data=[file name]``, that I've explored using to automatically download and start up the Docker container, but that has generally not worked reliably.
   
2. You need to open the 8888 (and 8787 if using Dask) ports for your instance, or ssh tunnel to access them.
   If you want to do this manually using a terminal, you can use the commands:
   ```
   gcloud compute firewall-rules create jupyter-notebook --allow https:8888
   gcloud compute firewall-rules create dask-webinterface --allow https:8787
   ```
   Note that these will create these firewall rules for your entire Google Cloud project. If you want to be more selective, you can do them just for an instance. (``gcloud compute firewall-rules create jupyter-notebook --allow https:8888 --source-tags=[instance-name]``).
     
3. ssh into your instance. You can use the browser-based shell or the gcloud command, ``gcloud compute ssh [instance-name]``.

4. Docker is already installed, so to start the container, you need to run the command:
```
docker run ckemere/jupyter -p 8888:8888 -p 8787:8787
```
The ``-p `` flags will expose the jupyter notebook ports and dask console ports from the container to the instance.

5. Now, find your instance's IP address from the web console, and load it into your browser ``https://instance-ip:8888`` and you should see a jupyter notebook.

6. In my typical workflow, I then use the jupyter terminal interface to clone whatever repository has my analysis scripts into the container. In the example notebooks, you'll also see how to use the (gcsfs)[http://gcsfs.readthedocs.io/en/latest/] package to load data stored in google cloud data buckets. You can upload data to these using a drag and drop web interface through the cloud console.

## Option 2 - Startup a container using the _alpha_ direct container option in google cloud.
You have to register for access to this, but it lets you just start a container during instance boot up. I've mainly done this in the web console. Once you've been granted access, a check box that says "Deploy a container image to this VM instance." will appear and you can chose that and then type in ``ckemere/jupyter`` as the container name. In this option, the container automatically exposes ports to the main network, so you don't need to worry about the ``-p`` option. You will, however, need to open your firewall if you want to access your notebooks directly (vs. an ssh tunnel).

## Option 3 - Startup a container and a dask scheduler using the Container Engine, rather than the Compute Engine
See the (dask-kubernetes)[https://github.com/dask/dask-kubernetes] repository for more information about this. Note that I've also created a version of the dask-kubernetes Docker image that has (nelpy)[https://github.com/nelpy/nelpy] installed. It's called
``ckemere/dask-kubernetes``. You can see those Docker files in my (fork of the repository)[https://github.com/ckemere/dask-kubernetes].

     


