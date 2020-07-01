# Smart camera for smart homes
## Environment
If you are already using a Ubuntu 16.04 skip the tutorial to Git Repository

## Development Environment 

To execute this paper is needed some tools:

* Virtual Machine
* OpenCV 3.3
* imutils python library
* A camera for the computer
* MQTT Broker
* paho mqtt library

First is needed to download a virtual machine, the tool used at this paper was found in that link: <https://www.virtualbox.org/>

After downloading and installing the virtual machine, is also needed to instal a update package to allow the virtual machine to have access to the integrated camera (for notebook users). to download this package click at this link: <https://download.virtualbox.org/virtualbox/6.1.10/Oracle_VM_VirtualBox_Extension_Pack-6.1.10.vbox-extpack>

The version to be downloaded MUST be the same of the virtual machine. At this point version 6.1.10 for the virtual machine and the update package. 

After download the package, open the virtual box, go at 'File' tab, click at preferences, select extensions and click at the add button, search the downloaded package and install it.

### Virtual machine

The next step is Download a linux image, the used for this paper was ubuntu 16.04, and install the image at the virtual box. The linux image can be found here: https://releases.ubuntu.com/16.04/

First of all, with the virtual machine opened, go to the devices tab, click at the webcam slot and allow the virtual machine to have access to the camera.

### Git Repository

At the ubuntu 16.04 environment, download the repository, if you have not git ate the machine, run:
```
sudo apt install git
```
Then, clone the repository:
```
git clone https://github.com/pablogurjao/ReproducibilityPaper.git
```

### OpenCV

It is needed to install the openCV.

It is a free library used for computational vision. The openCV version must be 3.3 or newer.

If you have not pip3 installed at the machine, run:

```
sudo apt install python3-pip
```

To install the openCV run:
```
pip3 install numpy
pip3 install opencv-python
```

### imutils

To use imutils library, it will be needed the instalation.

```
pip3 install imutils
```
### TKinter lib
This lib is necessary to run the graphical interface:
```
sudo apt-get install python3-tk
```
### MQTT 
to run the mqtt brocker is needed to run:
```
sudo apt-get install mosquitto mosquitto-clients
```
to use the mqtt at the python, it is needed to run:
```
pip3 install paho-mqtt
```

To the MQTT work properly, you need to check the ip of your machine. Go to the linux terminal and type:

```
ifconfig
```
![IP](https://github.com/pablogurjao/ReproducibilityPaper/blob/master/ip.png)

And copy your ip, as I am connected from wi-fi, mt ip is 127.0.0.1

with your ip paste this ip at the field "hostname" in the real time detect object file and at the "hostname" from them app.py file.

## Methods

To realize this experiments, two files at the repository must be executed, The "app.py" file, that create a graphical interface to show the status of the received command and the real_time_object_detection.py is the file that's open the camera visualization window and run the detection of the objects, to create a scenario and send a command for the "app.py" file.

With all resources defined and explained, the files must be run. Execute the app.py file frist.

Go to the directory that the repository was cloned and execute the app.py file.
```
cd USERPATH/ReproducibilityPaper/real-time-object-detection/

python3 app.py
```

After that, open the real_time_object_detection.py file, with a prediction precision of 70%:
```
python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --confidence 0.7
```



