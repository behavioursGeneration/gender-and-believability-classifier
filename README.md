# gender-and-believability-classifier

## Gender classifier 
This classifier is used to determine whether the speaker's gender is easily identifiable in the non-verbal behaviors transmitted. 

## Believability
This classifier/discriminator of generated vs. real behavior is trained on a GAN-type model of non-verbal behavior generation. The discriminator parameters of this GAN are extracted and used here to discriminate the believability of behavior files. 

We get 1 if the classifier thinks the behavior comes from a real behavior and 0 if the classifier thinks the behavior comes from a generated behavior.
