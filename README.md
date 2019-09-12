(note - still rewriting readme)

## NLP Machine Learning Bible Visualizer

**Problem:**
Religious texts tend to be very cryptic and often need good vizualizations to aid the reader. Have good visuals aids with improving the readers memory as well as helping readers (such as children) understand the passages better

**Solution:**
Use Machine Learning to find topic words in bible passages and visualize the passage using pictures instead.

**Implementation:**
Used LDA to data mine bible passages to train a model on the entire book of Revelations then use the model to predict important topic words in new
passages similar to passages in revelation (or from revelation itself) then visualize these images using an image gallery that allows users
to swipe through images like swiping through polaroid cameras on a table

![Screenshot](https://github.com/msimbao/bible_lda_visualizer/blob/master/screen.gif)



**Things I Learnt**

*Web Scrapping
*Probabilistic Models
*NLP and Text Mining

**Getting Started**

###What is LDA?
Latent Dirichlet Allocation (LDA) is a “generative probabilistic model” of a collection of composites made up of parts. Its uses include Natural Language Processing (NLP) and topic modelling, among others.

###Why use LDA?
If you view the number of topics as a number of clusters and the probabilities as the proportion of cluster membership, then using LDA is a way of soft-clustering your composites and parts.
Contrast this with say, k-means, where each entity can only belong to one cluster (hard-clustering). LDA allows for ‘fuzzy’ memberships. This provides a more nuanced way of recommending similar items, finding duplicates, or discovering user profiles/personas.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

* Kivy
* genisim
* google_images_download
* Beautiful Soup 4

### Installing

![Screenshot](https://github.com/msimbao/bible_lda_visualizer/blob/master/Screen2.png)

To install the prerequisites, you can go to their project pages to install them properly. The basic installation that I use is simply:

```
#Kivy needed to run GUI
sudo pip3 install kivy

#genisim for LDA
sudo pip3 install genisim

#google_images_download for getting Images
sudo pip3 install google_images_download

#beautiful soup 4 for webscrapping
sudo pip3 install bs4
```

There may also be additional packages depending on how your personal system is set up and you may be asked to install them as you go.

### Running the Program

To Run the Program, simply Clone the Repo and change directory to the repository's main file.

Then in the terminal type:

```
sudo python3 main.py
```

Below Is a picture showing the GUI Starting if everything has gone well.


### Changing the Training Passage:

To Change the training passage, open revelation.txt and replace the text inside with the document you want the LDA model to train off of.

```
Give an example
```
### Changing the Test Passage:

For this example, i used urls instead of direct copied and pasted text. So you will need a website with bible passages or 
text from whatever source you want to use.

I get my texts from [Bible GateWay](https://www.biblegateway.com/passage/?search=Revelation+11&version=NIV)

Open lda.py and scroll to the bottom. Change the string of the url variable to whatever website you want to use

## Built With

* [google_image_download](https://github.com/kootenpv/whereami) - To Get Images
* [genism](https://pypi.org/project/soundcloud-lib/) - To make LDA model
* [kivy](https://kivy.org/) - Used to generate GUI 
* [bs4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Used to Scrape Web


## Authors

* **Mphatso Simbao** - *Initial work* 

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE

## Acknowledgments

* https://lettier.com
* Ian Murphy
