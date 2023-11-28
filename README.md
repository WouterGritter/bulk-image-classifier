This is a small project that aims to process all images recursively in a certain directory (in my case on a NAS) and identify objects on it using YOLO. This output is
then stored, and can be used to filter through these images (eg. search for all images containing a cat and a dog, or search for images with 3 or more people in it).

In the future I plan to store the results in a database and make a service that repeatedly looks for new/modified images in the directory to (re)process them. This
makes it possible to develop a small front-end application that allows you to search through your images based on keywords, a self-hosted alternative to Google Photos which
provides a similar feature.

The large file `yolov7.pt` is omitted in this git repository, so if you want to run the project you will have to download this file yourself.
