## Hidden Footprints: Learning Contextual Walkability from 3D Human Trails
Jin Sun,	Hadar Averbuch-Elor,	Qianqian Wang, and	
Noah Snavely

European Conference on Computer Vision (ECCV) 2020

[Project Website](https://www.cs.cornell.edu/~jinsun/hidden_footprints.html "Project")

This repository contains code for hidden footprints propagation on Waymo data, as described in the paper. In addition, we provide a pretrained model to predict walkability on street scenes.

![Teaser](https://www.cs.cornell.edu/~jinsun/hiddenfootprints/results.png)


## Requirment

* Waymo Open Dataset [code base](https://github.com/waymo-research/waymo-open-dataset "Waymo").
* Tensorflow for reading Waymo data.
* PyTorch for running pretrained models.
* OpenCV for image handling.
* Numpy.


## Demo

First download Waymo data to your local directory. Extract the zip files to get *.tfrecord files.

Open demo.ipynb for an example on how to:
1) Load Waymo record file for images, labels, and camera parameters; 
2) Propagate hidden foorptins from all frames in a sequence (segment) to a reference frame;
3) Run a pretrained walkability prediction model on the reference frame image.

Pretrained model can be downloaded from Google Drive [here](https://drive.google.com/file/d/1Poub4YbK4Vl-iP64atTI3xsvt12eAE6Q/view?usp=sharing "Pretrained Model"). Put it in the root directory.

## License
This repository is released under the [Apache 2.0 license](LICENSE).

## Citation
```BibTeX
@InProceedings{hiddenfootprints2020eccv,
title={Hidden Footprints: Learning Contextual Walkability from 3D Human Trails},
author={Jin Sun and Hadar Averbuch-Elor and Qianqian Wang and Noah Snavely},
booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
month={August},
year={2020} 
}
```
