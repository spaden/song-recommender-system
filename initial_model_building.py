# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:20:00 2023

@author: KalyanRuchiPC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

import nltk

nltk.download('all')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random
import re

from keras.models import Model, load_model
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Flatten

reg = re.compile(r'[a-zA-Z]')



print(stopwords)

stop_list =['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stemming = PorterStemmer()


print(stop_list)

#df = pd.read_json('quotes.json')

df2 = pd.read_csv('tweet_emotions.csv')

print(set(df2['sentiment']))

# df['mod_Quote'] = df['Quote']
# df.drop_duplicates(subset ="Quote", keep = 'first', inplace = True)


# print(set(df['Category']))

# happy_df = df[df['Category'].isin(['humor', 'positive', 'romance', 'friendship', 'love', 'happiness'])]

# happy_df = happy_df['mod_Quote']

# happy_df = pd.DataFrame(happy_df)

# happy_songs = [
#         "It might seem crazy what I am 'bout to say Sunshine, she's here, you can take a break I'm a hot air balloon that could go to space With the air, like I don't care, baby by the way Huh Because I'm happy Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Here come bad news talking this and that Yeah Well give me all you got, don't hold back Yeah Well I should probably warn you I'll be just fine Yeah No offense to you don't waste your time Here's why Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Uh, bring me down Can't nothing, bring me down My level's too high to bring me down Can't nothing, bring me down, I said Bring me down, can't nothing Bring me down My level's too high to bring me down Can't nothing, bring me down, I said Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you Because I'm happy Clap along if you feel like that's what you wanna do Uh, bring me down Happy, happy, happy, happy Can't nothing Happy, happy, happy, happy Bring me down, my level's too high To bring me down Happy, happy, happy, happy Can't nothing Happy, happy, happy, happy Bring me down, I said Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you ayy, ayy, ayy Because I'm happy Clap along if you feel like that's what you wanna do Clap along if you feel like a room without a roof Because I'm happy Clap along if you feel like happiness is the truth Because I'm happy Clap along if you know what happiness is to you hey Because I'm happy Clap along if you feel like that's what you wanna do Come on",
#         "Oh! Ohhhh, yeeeh I used to think maybe you loved me now baby I'm sure And I just can't wait till the day when you knock on my door Now everytime I go for the mailbox, gotta hold myself down 'Cause I just can't wait till you write me you're coming around I'm walking on sunshine, wooah I'm walking on sunshine, woooah I'm walking on sunshine, woooah And don't it feel good! Hey, alright now And dont it feel good! Hey yeah I used to think maybe you loved me, now I know that it's true And I don't want to spend my whole life, just waiting for you Now I don't want u back for the weekend Not back for a day, no no no I said baby I just want you back and I want you to stay Woah yeah! I'm walking on sunshine, wooah I'm walking on sunshine, woooah I'm walking on sunshine, woooah And don't it feel good!! Hey, alright now And don't it feel good!! Yeah, oh yeah And don't it feel good!! Walking on sunshine Walking on sunshine I feel alive, I feel the love, I feel the love that's really real I feel alive, I feel the love, I feel the love that's really real I'm on sunshine baby oh, oh yeah I'm on sunshine baby oh I'm walking on sunshine, wooah I'm walking on sunshine, wooah I'm walking on sunshine, woooah And don't it feel good Hey, alright now And don't it feel good I'll say it, say it, say it again now And don't it feel good!! Ohhhh, yeahhhh And don't it feel good Now don't it, don't it, don't it, don't it, don't it, don't it feel good I'll say it, say it, say it again now And don't feel good Now don't it, don't it, don't it, don't it, don't it, don't it feel good Tell me, tell me, tell me again now And don't it feel good Ohhhh, yeahhhh And don't it feel good Oh don't it feel good, don't it feel good Now don't it feel good? Oh yeah, oh yeah, oh yeah Don't it feel good",
#         "I'm coming out I'm coming I'm coming out I'm coming out I'm coming out I'm coming out I'm coming out I want the world to know Got to let it show I'm coming out I want the world to know Got to let it show There's a new me coming out And I just have to live And I want to give I'm completely positive I think this time around I am gonna do it Like you never knew it Oh, I'll make it through The time has come for me To break out of the shell I have to shout That I am coming out I'm coming out I want the world to know I got to let it show I'm coming out I want the world to know I got to let it show I'm coming out coming I want the world to know out Got to let it show I'm coming I'm coming out coming I want the world to know out Got to let it show I've got to show the world All that I want to be And all my abilities There's so much more to me Somehow, I'll have to make them Just understand I got it well in hand And, oh, how I've planned I'm spreadin' love There is no need to fear And I just feel so good Every time I hear I'm coming out I want the world to know I got to let it show I'm coming out coming I want the world to know out Got to let it show I'm coming I'm coming out coming I want the world to know out Got to let it show I'm coming out I want the world to know Got to let it show I'm coming out I want the world to know Got to let it show want the world to know I got to let it show I'm coming out I want the world to know Got to let it show I'm coming out I'm coming I want the world to know out Got to let it show I want the world to know, gotta let it show I'm coming out I'm, I'm, I'm I want the world to know I'm coming out Got to let it show watch out, I'm coming out I'm coming out I want the world to know I got to let it show I'm coming out coming, coming out I want the world to know Got to let it show want the world to know Got to let it show I'm, I'm coming out I have to shout that I'm comin' out I want the world to know, I gotta let it show I'm coming, I'm coming out",
#         "Ah, yeah, ah, yeah I got this feelin' inside my bones It goes electric, wavy when I turn it on All through my city, all through my home We're flyin' up, no ceilin', when we in our zone I got that sunshine in my pocket Got that good soul in my feet I feel that hot blood in my body when it drops ooh I can't take my eyes up off it, movin' so phenomenally Room on lock, the way we rock it, so don't stop And under the lights when everything goes Nowhere to hide when I'm gettin' you close When we move, well, you already know So just imagine, just imagine, just imagine Nothin' I can see but you when you dance, dance, dance Feel a good, good creepin' up on you So just dance, dance, dance, come on All those things I shouldn't do But you dance, dance, dance And ain't nobody leavin' soon, so keep dancin' I can't stop the feelin' So just dance, dance, dance I can't stop the feelin' So just dance, dance, dance, come on Ooh, it's something magical It's in the air, it's in my blood, it's rushin' on rushin' on I don't need no reason, don't need control need control I fly so high, no ceiling, when I'm in my zone 'Cause I got that sunshine in my pocket Got that good soul in my feet I feel that hot blood in my body when it drops ooh I can't take my eyes up off it, moving so phenomenally Room on lock, the way we rock it, so don't stop stop, stop, stop Under the lights when everything goes Nowhere to hide when I'm gettin' you close When we move, well, you already know So just imagine, just imagine, just imagine Nothing I can see but you when you dance, dance, dance Feel a good, good, creepin' up on you So just dance, dance, dance, come on All those things I shouldn't do But you dance, dance, dance And ain't nobody leavin' soon, so keep dancin' I can't stop the feelin' So just dance, dance, dance I can't stop the feelin' So just dance, dance, dance I can't stop the feelin' So just dance, dance, dance I can't stop the feelin' yeah So keep dancin', come on Oh, yeah, yeah I can't stop the, I can't stop the I can't stop the, I can't stop the I can't stop the feelin' Nothin' I can see but you when you dance, dance, dance I can't stop the feelin' Feel the good, good, creepin' up on you So just dance, dance, dance, come on I can't stop the feelin' All those things I shouldn't do But you dance, dance, dance dance, dance, dance I can't stop the feelin' And ain't nobody leavin' soon, so keep dancin' Everybody sing I can't stop the feelin' Got this feeling in my body I can't stop the feelin' Got this feeling in my body I can't stop the feelin' Wanna see you move your body I can't stop the feelin' Got this feelin' in my body Break it down Got this feelin' in my body ah Can't stop the feelin' Got this feelin' in my body, come on ooh",
#         "I come home, in the mornin' light My mother says, When you gonna live your life right? Oh momma dear, we're not the fortunate ones And girls, they wanna have fun Oh girls just wanna have fun The phone rings, in the middle of the night My father yells, What you gonna do with your life? Oh daddy dear, you know you're still number one But girls, they wanna have fun Oh girls just wanna have That's all they really want Some fun When the workin' day is done Oh girls, they wanna have fun Oh girls just wanna have fun girls, they want Wanna have fun, girls Wanna have Some boys take a beautiful girl And hide her away from the rest o' the world I wanna be the one to walk in the sun Oh girls, they wanna have fun Oh girls just wanna have That's all they really want Is some fun When the workin' day is done Oh girls, they wanna have fun Oh girls just wanna have fun girls, they want Wanna have fun, girls Wanna have They just want, they just wanna girls They just want, they just wanna girls just wanna have fun Oh girls, girls just wanna have fun Just want, they just wanna They just wanna, they just wanna girls They just want, they just wanna girls just wanna have fun Oh girls, girls just wanna have fun When the workin' When the workin' day is done Oh, when the workin' day is done Oh, girls, girls just wanna have fun Everybody Huh, huh They just want, they just wanna girls They just want, they just wanna girls just wanna have fun Oh, girls, girls just wanna have fun They just wanna, they just wanna when the workin' When the working day is done they just want, they just wanna Oh, when the working day is done girls, girls just wanna have fun Oh girls, girls just wanna have fun",
#         "Here's a little song I wrote You might want to sing it note for note Don't worry, be happy In every life we have some trouble But when you worry, you make it double Don't worry, be happy Don't worry, be happy now Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Don't worry, be happy Ooh, ooh-ooh-ooh-ooh-ooh-ooh-ooh-ooh Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Don't worry, be happy Ain't got no place to lay your head Somebody came and took your bed Don't worry, be happy The landlord say your rent is late He may have to litigate Don't worry, be happy look at me, I'm happy Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Hey I give you my phone number When you worry, call me, I make you happy Ooh, ooh-ooh-ooh-ooh-ooh-ooh-ooh-ooh Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Ain't got no cash, ain't got no style Ain't got no gal to make you smile But don't worry, be happy 'Cause when you worry your face will frown And that will bring everybody down So don't worry, be happy Don't worry, be happy now Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Don't worry, be happy Ooh, ooh-ooh-ooh-ooh-ooh-ooh-ooh-ooh Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Don't worry, be happy Now there is this song I wrote I hope you learned it note for note, like good little children Dn't worry, be happy Now listen to what I said, in your life expect some trouble But when you worry, you make it double But don't worry, be happy, be happy now Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Don't worry, be happy Ooh, ooh-ooh-ooh-ooh-ooh-ooh-ooh-ooh Don't worry Ooh-ooh-ooh-ooh Be happy Ooh-ooh-ooh Don't worry, be happy Ooh, ooh-ooh-ooh-ooh-ooh-ooh-ooh-ooh don't worry, don't worry Ooh-ooh-ooh-ooh Don't do it, be happy Ooh-ooh-ooh Put a smile in your face, don't bring everybody down like this Ooh, ooh-ooh-ooh-ooh-ooh-ooh-ooh-ooh Don't worry Ooh-ooh-ooh-ooh It will soon pass, whatever it is Ooh-ooh-ooh don't worry, be happy Ooh, ooh-ooh-ooh-ooh-ooh-ooh-ooh-ooh I'm not worried Ooh-ooh-ooh-ooh I'm happy",
#         "It feels like a perfect night To dress up like hipsters And make fun of our exes Ah-ah, ah-ah It feels like a perfect night For breakfast at midnight To fall in love with strangers Ah-ah, ah-ah Yeah, we're happy, free, confused and lonely at the same time It's miserable and magical, oh yeah Tonight's the night when we forget about the deadlines It's time, oh-oh I don't know about you But I'm feeling 22 Everything will be alright if You keep me next to you You don't know about me But I'll bet you want to Everything will be alright if We just keep dancing like we're 22, 22 It seems like one of those nights This place is too crowded Too many cool kids Who's Taylor Swift anyway? Ew Ah-ah, ah-ah It seems like one of those nights We ditch the whole scene And end up dreamin' instead of sleeping, yeah We're happy, free, confused and lonely in the best way It's miserable and magical, oh yeah Tonight's the night when we forget about the heartbreaks It's time, oh-oh I don't know about you But I'm feeling 22 Everything will be alright if You keep me next to you You don't know about me But I'll bet you want to Everything will be alright if We just keep dancing like we're 22 Oh, oh, oh, oh 22 I don't know about you 22, 22 It feels like one of those nights We ditch the whole scene It feels like one of those nights We won't be sleeping It feels like one of those nights You look like bad news I gotta have you I gotta have you Ooh, ooh, yeah, yeah I don't know about you But I'm feeling 22 Everything will be alright if Ooh You keep me next to you You don't know about me But I'll bet you want to Everything will be alright if We just keep dancing like we're 22 22 Dancing like 22 Yeah, yeah 22 Yeah, yeah, yeah It feels like one of those nights We ditch the whole scene It feels like one of those nights We won't be sleeping It feels like one of those nights You look like bad news I gotta have you I gotta have you",
#         "I do my hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin'? Feeling good as hell Woo child, tired of the bullshit Go on dust your shoulders off, keep it moving Yes Lord, tryna get some new shit In there, swimwear, going to the pool shit Come now, come dry your eyes You know you a star, you can touch the sky I know that it's hard but you have to try If you need advice, let me simplify If he don't love you anymore Just walk your fine ass out the door I do my hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin'? Feeling good as hell Feeling good as hell Baby how you feelin'? Feeling good as hell Woo girl, need to kick off your shoes Got to take a deep breath, time to focus on you All the big fights, long nights that you been through I got a bottle of Tequila I been saving for you Boss up and change your life You can have it all, no sacrifice I know he did you wrong, we can make it right So go and let it all hang out tonight 'Cause he don't love you anymore So walk your fine ass out the door And do your hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin'? Feeling good as hell Listen, if he don't love you anymore Then walk your fine ass out the door And do your hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin'? Feeling good as hell Hair toss Check my nails Baby how you feelin' Feeling good as hell Feeling good as hell Baby how you feelin'? Feeling good as hell",
#         "You're the light, you're the night You're the color of my blood You're the cure, you're the pain You're the only thing I wanna touch Never knew that it could mean so much, so much You're the fear, I don't care 'Cause I've never been so high Follow me through the dark Let me take you past our satellites You can see the world you brought to life, to life So love me like you do, lo-lo-love me like you do Love me like you do, lo-lo-love me like you do Touch me like you do, to-to-touch me like you do What are you waiting for Fading in, fading out On the edge of paradise Every inch of your skin is a holy grail I've got to find Only you can set my heart on fire, on fire Yeah, I'll let you set the pace 'Cause I'm not thinking straight My head's spinning around I can't see clear no more What are you waiting for Love me like you do, lo-lo-love me like you do like you do Love me like you do, lo-lo-love me like you do Touch me like you do, to-to-touch me like you do What are you waiting for Love me like you do, lo-lo-love me like you do like you do Love me like you do, lo-lo-love me like you do yeah Touch me like you do, to-to-touch me like you do What are you waiting for I'll let you set the pace 'Cause I'm not thinking straight My head's spinning around I can't see clear no more What are you waiting for Love me like you do, lo-lo-love me like you do like you do Love me like you do, lo-lo-love me like you do yeah Touch me like you do, to-to-touch me like you do What are you waiting for Love me like you do, lo-lo-love me like you do like you do Love me like you do, lo-lo-love me like you do oh Touch me like you do, to-to-touch me like you do What are you waiting for",
#         "It's the strangest feeling Feeling this way for you There's something in the way you move Something in the way you move With you, I'm never healing It's heartache through and through There's something in the way you move I don't know what it is you do Not one bone in your body good enough for me But this heart is open, bloodstain on my sleeve When our eyes meet, I can only see the end But tonight I'm here, yours again But tonight I'm gonna lose it all Playing with fire, I was the first to fall Heart is sinking like a cannonball Baby, kill it, what you waiting for? Something in the way you move There's something in the way you do it Something in the way you move, oh-oh There's something in the way you move Something in the way you do it Something in the way you move, oh-oh Oh-oh, oh-oh-oh Oh-oh, oh-oh-oh There's an evil night air The stars don't shine tonight night There's something in the way you do There's something in the way you Push me closer, further Break me just enough Your lies always seem so true There's nothing left for me to lose There's not one thing I can do to change your ways But I can't sit back and take the lonely days When our eyes meet, I can only see the end And tonight the rain pours again But tonight I'm gonna lose it all Playing with fire, I was the first to fall Heart is sinking like a cannonball Baby, kill it, what you waiting for? Something in the way you move There's something in the way you do it Something in the way you move, oh-oh There's something in the way you move Something in the way you do it Something in the way you move, oh-oh Oh-oh, oh-oh-oh Oh-oh, oh-oh-oh Oh, oh, oh-oh Oh, oh, oh-oh Oh-oh, oh-oh-oh Oh-oh, oh-oh-oh Oh, oh, oh-oh But tonight I'm gonna lose it all Playing with fire, I was the first to fall Heart is sinking like a cannonball Baby, kill it, what you waiting for? Something in the way you move There's something in the way you do it Something in the way you move, oh-oh There's something in the way you move Something in the way you do it Something in the way you move, oh-oh Something in the way you move There's something in the way you do it Something in the way you move, oh-oh There's something in the way you move Something in the way you do it Something in the way you move, oh-oh",
#         "Hey, I was doing just fine before I met you I drink too much and that's an issue, but I'm okay Hey, you tell your friends it was nice to meet them But I hope I never see them again I know it breaks your heart Moved to the city in a broke-down car, and Four years, no calls Now you're looking pretty in a hotel bar And I, I, I, I, I can't stop No, I, I, I, I, I can't stop So, baby, pull me closer In the back seat of your Rover That I know you can't afford Bite that tattoo on your shoulder Pull the sheets right off the corner Of that mattress that you stole From your roommate back in Boulder We ain't ever getting older We ain't ever getting older We ain't ever getting older You look as good as the day I met you I forget just why I left you, I was insane Stay and play that Blink-182 song That we beat to death in Tucson, okay I know it breaks your heart Moved to the city in a broke-down car, and Four years, no call Now I'm looking pretty in a hotel bar And I, I, I, I, I can't stop No, I, I, I, I, I can't stop So, baby, pull me closer In the back seat of your Rover That I know you can't afford Bite that tattoo on your shoulder Pull the sheets right off the corner Of that mattress that you stole From your roommate back in Boulder We ain't ever getting older We ain't ever getting older We ain't ever getting older",
#         "Fighting flames of fire Hang onto burning wires We don't care anymore Are we fading lovers? We keep wasting colors Maybe we should let this go We're fallin' apart, still we hold together We've passed the end, so we chase forever 'Cause this is all we know This feeling's all we know I'll ride my bike up to the road Down the streets, right through the city I'll go everywhere you go From Chicago to the coast You tell me, Hit this and let's go Blow the smoke right through the window 'Cause this is all we know 'Cause this is all we know 'Cause this is all we know Never face each other One bed, different covers We don't care anymore Two hearts still beating On with different rhythms Maybe we should let this go We're fallin' apart, still we hold together We've passed the end, so we chase forever 'Cause this is all we know This feeling's all we know I'll ride my bike up to the road Down the streets, right through the city I'll go everywhere you go From Chicago to the coast You tell me, Hit this and let's go Blow the smoke right through the window 'Cause this is all we know 'Cause this is all we know 'Cause this is all we knowr",
#         "I've been reading books of old The legends and the myths Achilles and his gold Hercules and his gifts Spiderman's control And Batman with his fists And clearly I don't see myself upon that list But she said, Where'd you wanna go? How much you wanna risk? I'm not looking for somebody With some superhuman gifts Some superhero Some fairytale bliss Just something I can turn to Somebody I can kiss I want something just like this Doo-doo-doo, doo-doo-doo Doo-doo-doo, doo-doo Doo-doo-doo, doo-doo-doo Oh, I want something just like this Doo-doo-doo, doo-doo-doo Doo-doo-doo, doo-doo Doo-doo-doo, doo-doo-doo Oh, I want something just like this I want something just like this I've been reading books of old The legends and the myths The testaments they told The moon and its eclipse And Superman unrolls A suit before he lifts But I'm not the kind of person that it fits She said, Where'd you wanna go? How much you wanna risk? I'm not looking for somebody With some superhuman gifts Some superhero Some fairytale bliss Just something I can turn to Somebody I can miss I want something just like this I want something just like this Oh, I want something just like this Doo-doo-doo, doo-doo-doo Doo-doo-doo, doo-doo Doo-doo-doo, doo-doo-doo Oh, I want something just like this Doo-doo-doo, doo-doo-doo Doo-doo-doo, doo-doo Doo-doo-doo, doo-doo-doo Where'd you wanna go? How much you wanna risk? I'm not looking for somebody With some superhuman gifts Some superhero Some fairytale bliss Just something I can turn to Somebody I can kiss I want something just like this Oh, I want something just like this Oh, I want something just like this Oh, I want something just like this",
#         "I'll tell you a story before it tells itself I'll lay out all my reasons, you'll say that I need help We all got expectations and sometimes they go wrong But no one listens to me, so I put it in this song They tell me think with my head, not that thing in my chest They got their hands at my neck this time But you're the one that I want and if that's really so wrong Then they don't know what this feeling is like And I say yeah Yeah Yeah Yeah And I say yeah Yeah Yeah Yeah I'll tell them a story, they'll sit and nod their heads I'll tell you all my secrets and you tell all your friends Hold on to your opinions and stand by what you said stand by what you said In the end it's my decision, so it's my fault when it ends They tell me think with my head, not that thing in my chest They got their hands at my neck this time But you're the one that I want and if that's really so wrong Then they don't know what this feeling is like And I say yeah Yeah Yeah Yeah And I say yeah Yeah Yeah Yeah I'll tell you a story before it tells itself I'll lay out all my reasons, you'll say that I need help We all got expectations and sometimes they go wrong But no one listens to me, so I put it in this song They tell me think with my head, not that thing in my chest They got their hands at my neck this time But you're the one that I want and if that's really so wrong Then they don't know what this feeling is like My friends say no, no But they don't know, know Yeah, they don't know, know They don't know no no, know And I say no oh, oh And I say, yeah, yeah, yeah, yeah They don't know",
#         "When I think of my mother No one compares to her A love like no other She puts everyone else first And when I was younger I treated her the worst Never known someone stronger 'Cause damn it must've hurt I don't see you as much as I used to But if I did, I know what I would do I'd tell you I love you a million times Say that I'm sorry if I made you cry Could never be half the woman even if I tried But I'll try, I swear I'll try If someone puts me down, yeah, I know my worth All thanks to you, the lessons that I've learnt If I had to put it into words I think of an angel and all I see is her Hmm ooh All I see is her Uh uh hmm Your voice in my head and It tells me I'm beautiful And when I have children I'll pass on the things I was told I don't see you as much as I used to But if I did, I know what I would do I'd tell you I love you a million times Say that I'm sorry if I made you cry Could never be half the woman even if I tried But I'll try, I swear I'll try If someone puts me down, yeah, I know my worth It's all thanks to you, the lessons that I've learnt If I had to put it into words I think of an angel and all I see is her Hmm ooh All I see is her Uh uh hmm All I see is her",
#         "I will always remember The day you kissed my lips Light as a feather And it went just like this No, it's never been better Than the summer of 2002 ooh Uh, we were only eleven But acting like grown-ups Like we are in the present Drinking from plastic cups Singing, Love is forever and ever Well, I guess that was true Dancing on the hood in the middle of the woods On an old Mustang, where we sang Songs with all our childhood friends And it went like this, say Oops, I got 99 problems singing bye, bye, bye Hold up, if you wanna go and take a ride with me Better hit me, baby, one more time, uh Paint a picture for you and me On the days when we were young, uh Singing at the top of both our lungs Now we're under the covers Fast forward to eighteen We are more than lovers Yeah, we are all we need When we're holding each other I'm taken back to 2002 ooh Yeah Dancing on the hood in the middle of the woods On an old Mustang, where we sang Songs with all our childhood friends And it went like this, say Oops, I got 99 problems singing bye, bye, bye Hold up, if you wanna go and take a ride with me Better hit me, baby, one more time, uh Paint a picture for you and me On the days when we were young, uh Singing at the top of both our lungs On the day we fell in love Ooh ooh, ooh ooh On the day we fell in love Ooh ooh, ooh ooh Dancing on the hood in the middle of the woods On an old Mustang, where we sang Songs with all our childhood friends Oh, now Oops, I got 99 problems singing bye, bye, bye Hold up, if you wanna go and take a ride with me Better hit me, baby, one more time Paint a picture for you and me On the days when we were young Singing at the top of both our lungs On the day we fell in love Ooh ooh, ooh ooh On the day we fell in love Ooh ooh, ooh ooh On the day we fell in love Ooh ooh, ooh ooh On the day we fell in love Ooh ooh, ooh ooh On the day we fell in love, love, love"
# ]


# happy_songs = pd.DataFrame({'mod_Quote': happy_songs})

happy_tweet = df2[df2['sentiment'].isin(['love', 'happiness'])]['content']

happy_tweet = pd.DataFrame({'mod_Quote': happy_tweet})

happy_df = happy_tweet

happy_df['label'] = 0

#happy_df = happy_df.iloc[:2000]


#sad_df = df[df['Category'].isin(['hope', 'death', 'faith', 'god', 'purpose'])]['mod_Quote']


#sad_df = pd.DataFrame(sad_df)


# sad_songs = [
#     "Ooh-ooh-ooh Ah-ah-ah-ah-ah Brown guilty eyes and little white lies Yeah, I played dumb but I always knew That you'd talk to her, maybe did even worse I kept quiet so I could keep you And ain't it funny How you ran to her The second that we called it quits? And ain't it funny How you said you were friends? Now it sure as hell don't look like it You betrayed me And I know that you'll never feel sorry For the way I hurt, yeah You'd talk to her When we were together Loved you at your worst But that didn't matter It took you two weeks To go off and date her Guess you didn't cheat But you're still a traitor Now you bring her around Just to shut me down Show her off like she's a new trophy And I know if you were true There's no damn way that you Could fall in love with somebody that quickly Ain't it funny All the twisted games All the questions you used to avoid? Ain't it funny? Remember I brought her up And you told me I was paranoid You betrayed me And I know that you'll never feel sorry For the way I hurt, yeah You'd talk to her When we were together Loved you at your worst But that didn't matter It took you two weeks To go off and date her Guess you didn't cheat But you're still a traitor God, I wish that you had thought this through Before I went and fell in love with you Ah-ah-ah When she's sleeping in the bed we made Don't you dare forget about the way You betrayed me 'Cause I know that you'll never feel sorry For the way I hurt, yeah You'd talk to her When we were together You gave me your word But that didn't matter It took you two weeks To go off and date her Guess you didn't cheat But you're still You're still a traitor ah-ah-ah Yeah, you're still a traitor Ooh-ooh-ooh God, I wish that you had thought this through Before I went and fell in love with you",
#       "Everybody knows my name now But somethin' 'bout it still feels strange Like lookin' in a mirror, tryna steady yourself And seein' somebody else And everything is not the same now It feels like all our lives have changed Maybe when I'm older, it'll all calm down But it's killin' me now What if you had it all But nobody to call? Maybe then you'd know me 'Cause I've had everything But no one's listening And that's just lonely I'm so lonely Lonely Everybody knows my past now Like my house was always made of glass And maybe that's the price you pay For the money and fame at an early age And everybody saw me sick And it felt like no one gave They criticized the things I did as an idiot kid What if you had it all But nobody to call? Maybe then you'd know me 'Cause I've had everything But no one's listening And that's just lonely I'm so lonely Lonely I'm so lonely Lonely",
#       "Wish I could, I could've said goodbye I would've said what I wanted to Maybe even cried for you If I knew it would be the last time I would've broke my heart in two Tryna save a part of you Don't wanna feel another touch Don't wanna start another fire Don't wanna know another kiss No other name fallin' off my lips Don't wanna give my heart away To another stranger Or let another day begin Won't even let the sunlight in No, I'll never love again I'll never love again, oh, oh, oh, oh When we first met I never thought that I would fall I never thought that I'd find myself Lying in your arms, mm, mm And I wanna pretend that it's not true Oh, baby, that you're gone 'Cause my world keeps turnin', and turnin', and turnin' And I'm not movin' on Don't wanna feel another touch Don't wanna start another fire Don't wanna know another kiss No other name fallin' off my lips Don't wanna give my heart away To another stranger Or let another day begin Won't even let the sunlight in No, I'll never love I don't wanna know this feelin' Unless it's you and me I don't wanna waste a moment, ooh And I don't wanna give somebody else the better part of me I would rather wait for you, ooh Don't wanna feel another touch Don't wanna start another fire Don't wanna know another kiss Baby, unless they are your lips Don't wanna give my heart away To another stranger Don't let another day begin Won't let the sunlight in Oh, I'll never love again Never love again Never love again Oh, I'll never love again",
#       "We found each other I helped you out of a broken place You gave me comfort But falling for you was my mistake I put you on top, I put you on top I claimed you so proud and openly And when times were rough, when times were rough I made sure I held you close to me So call out my name call out my name Call out my name when I kiss you so gently I want you to stay I want you to stay I want you to stay, even though you don't want me Girl, why can't you wait? why can't you wait, baby? Girl, why can't you wait 'til I fall out of love? Won't you call out my name? call out my name Girl, call out my name, and I'll be on my way and I'll be on my I said I didn't feel nothing baby, but I lied I almost cut a piece of myself for your life Guess I was just another pit stop 'Til you made up your mind You just wasted my time You're on top, I put you on top I claimed you so proud and openly, babe And when times were rough, when times were rough I made sure I held you close to me So call out my name call out my name, baby So call out my name when I kiss you So gently, I want you to stay I want you to stay I want you to stay even though you don't want me Girl, why can't you wait? girl, why can't you wait 'til I Girl, why can't you wait 'til I fall out of loving? Babe, call out my name say call out my name, baby Girl, call out my name, and I'll be on my way, girl I'll be on my On my way, on my way On my way, on my way, ooh On my way, on my way, on my way On my way, on my way, on my way On my",
#       "Mmh-mh I don't know when you're sad I can't tell when you're mad You've never been vulnerable I believed you could change But you're still the same And I'm still invisible I keep secrets from my family All the ways you been controlling me When it hurt, I didn't make a sound All the drama that you put us through How amazing that it's never you Who lit the fire, burnt the whole house down 'Cause nobody taught you how to cry But somebody showed you how to lie All of the feelings you don't show Are all of the reasons to let go I know I got nothing left I know I got nothing left I get that it's deep You've never been free You've never been satisfied And you blame that on me The pain you can't see The anger you hold inside I keep secrets from my friends at home So embarrassed, I've been so alone God I'm over me protecting you 'Cause nobody taught you how to cry But somebody showed you how to lie All of the feelings you don't show Are all of the reasons to let go I know I got nothing left I know I got nothing left, no I know I got nothing left, mmh-mh I know I got nothing left, mmh-mmh",
#       "I walked through the door with you The air was cold But something about it felt like home somehow And I, left my scarf there at your sister's house And you've still got it in your drawer even now Oh, your sweet disposition And my wide-eyed gaze We're singing in the car, getting lost upstate Autumn leaves falling down like pieces into place And I can picture it after all these days And I know it's long gone and that magic's not here no more And I might be okay but I'm not fine at all 'Cause there we are again on that little town street You almost ran the red 'cause you were lookin' over at me Wind in my hair, I was there I remember it all too well Photo album on the counter Your cheeks were turning red You used to be a little kid with glasses in a twin-sized bed And your mother's telling stories 'bout you on the tee-ball team You told me 'bout your past thinking your future was me And I know it's long gone and there was nothing else I could do And I forget about you long enough to forget why I needed to 'Cause there we are again in the middle of the night We're dancing 'round the kitchen in the refrigerator light Down the stairs, I was there I remember it all too well, yeah And maybe we got lost in translation Maybe I asked for too much But maybe this thing was a masterpiece 'til you tore it all up Running scared, I was there I remember it all too well And you call me up again just to break me like a promise So casually cruel in the name of being honest I'm a crumpled up piece of paper lying here 'Cause I remember it all, all, all Too well Time won't fly, it's like I'm paralyzed by it I'd like to be my old self again But I'm still trying to find it After plaid shirt days and nights when you made me your own Now you mail back my things and I walk home alone But you keep my old scarf from that very first week 'Cause it reminds you of innocence And it smells like me You can't get rid of it 'Cause you remember it all too well, yeah 'Cause there we are again when I loved you so Back before you lost the one real thing you've ever known It was rare, I was there, I remember it all too well Wind in my hair, you were there, you remember it all Down the stairs, you were there, you remember it all It was rare, I was there, I remember it all too well",
#       "You promised the world and I fell for it I put you first and you adored it Set fires to my forest And you let it burn Sang off-key in my chorus 'Cause it wasn't yours I saw the signs and I ignored it Rose-colored glasses all distorted Set fire to my purpose And I let it burn You got off on the hurtin' When it wasn't yours, yeah We'd always go into it blindly I needed to lose you to find me This dancing was killing me softly I needed to hate you to love me, yeah To love, love, yeah To love, love, yeah To love, yeah I needed to lose you to love me, yeah To love, love, yeah To love, love, yeah To love, yeah I needed to lose you to love me I gave my all and they all know it Then you tore me down and now it's showing In two months, you replaced us Like it was easy Made me think I deserved it In the thick of healing, yeah We'd always go into it blindly I needed to lose you to find me This dancing was killing me softly I needed to hate you to love me, yeah To love, love, yeah To love, love, yeah To love, yeah I needed to lose you to love me, yeah To love, love, yeah To love, love, yeah To love, yeah I needed to lose you to love me You promised the world and I fell for it I put you first and you adored it Set fires to my forest And you let it burn Sang off-key in my chorus To love, love, yeah To love, love, yeah To love, yeah I needed to hate you to love me, yeah To love, love, yeah To love, love, yeah To love, yeah I needed to lose you to love me To love, love, yeah To love, love, yeah To love, yeah And now the chapter is closed and done To love, love, yeah To love, love, yeah To love, yeah And now it's goodbye, it's goodbye for us",
#       "Well, you only need the light when it's burning low Only miss the sun when it starts to snow Only know you love her when you let her go Only know you've been high when you're feeling low Only hate the road when you're missing home Only know you love her when you let her go And you let her go Staring at the bottom of your glass Hoping one day you'll make a dream last But dreams come slow, and they go so fast You see her when you close your eyes Maybe one day, you'll understand why Everything you touch surely dies But you only need the light when it's burning low Only miss the sun when it starts to snow Only know you love her when you let her go Only know you've been high when you're feeling low Only hate the road when you're missing home Only know you love her when you let her go Staring at the ceiling in the dark Same old empty feeling in your heart 'Cause love comes slow, and it goes so fast Well, you see her when you fall asleep But never to touch and never to keep 'Cause you loved her too much, and you dived too deep Well, you only need the light when it's burning low Only miss the sun when it starts to snow Only know you love her when you let her go Only know you've been high when you're feeling low Only hate the road when you're missing home Only know you love her when you let her go And you let her go Oh, oh, mm, oh And you let her go Oh, oh, uh, uh Well, you let her go 'Cause you only need the light when it's burning low Only miss the sun when it starts to snow Only know you love her when you let her go Only know you've been high when you're feeling low Only hate the road when you're missing home Only know you love her when you let her go 'Cause you only need the light when it's burning low Only miss the sun when it starts to snow Only know you love her when you let her go Only know you've been high when you're feeling low Only hate the road when you're missing home Only know you love her when you let her go",
#       "It's been a long day without you, my friend And I'll tell you all about it when I see you again We've come a long way from where we began Oh, I'll tell you all about it when I see you again When I see you again Damn, who knew? All the planes we flew, good things we been through That I'd be standing right here talking to you 'Bout another path, I know we loved to hit the road and laugh But something told me that it wouldn't last Had to switch up, look at things different, see the bigger picture Those were the days, hard work forever pays Now I see you in a better place see you in a better place Uh How can we not talk about family when family's all that we got? Everything I went through, you were standing there by my side And now you gon' be with me for the last ride It's been a long day without you, my friend And I'll tell you all about it when I see you again I'll see you again We've come a long way yeah, we came a long way From where we began you know we started Oh, I'll tell you all about it when I see you again I'll tell you When I see you again First, you both go out your way and the vibe is feeling strong And what's small turned to a friendship, a friendship turned to a bond And that bond will never be broken, the love will never get lost The love will never get lost And when brotherhood come first, then the line will never be crossed Established it on our own when that line had to be drawn And that line is what we reached, so remember me when I'm gone Remember me when I'm gone How can we not talk about family when family's all that we got? Everything I went through you were standing there by my side And now you gon' be with me for the last ride So let the light guide your way, yeah Hold every memory as you go And every road you take Will always lead you home, home It's been a long day without you, my friend And I'll tell you all about it when I see you again We've come a long way from where we began Oh, I'll tell you all about it when I see you again When I see you again When I see you again yeah, uh See you again yeah, yeah, yeah When I see you again",
#       "I still remember the look on your face Lit through the darkness at 1:58 The words that you whispered for just us to know You told me you loved me So why did you go away? Away I do recall now the smell of the rain Fresh on the pavement, I ran off the plane That July ninth, the beat of your heart It jumps through your shirt I can still feel your arms But now I'll go Sit on the floor wearing your clothes All that I know is I don't know How to be something you miss I never thought we'd have a last kiss Never imagined we'd end like this Your name, forever the name on my lips I do remember the swing of your step The life of the party, you're showing off again And I'd roll my eyes and then you'd pull me in I'm not much for dancing, but for you, I did Because I love your handshake, meeting my father I love how you walk with your hands in your pockets How you'd kiss me when I was in the middle of saying something There's not a day I don't miss those rude interruptions And I'll go Sit on the floor wearing your clothes All that I know is I don't know How to be something you miss I never thought we'd have a last kiss Never imagined we'd end like this Your name, forever the name on my lips Ooh So I'll watch your life in pictures like I used to watch you sleep And I feel you forget me like I used to feel you breathe And I'll keep up with our old friends just to ask them how you are Hope it's nice where you are And I hope the sun shines and it's a beautiful day And something reminds you you wish you had stayed You can plan for a change in the weather and time But I never planned on you changing your mind So I'll go Sit on the floor wearing your clothes All that I know is I don't know How to be something you miss I never thought we'd have a last kiss Never imagined we'd end like this Your name, forever the name on my lips Just like our last kiss Forever the name on my lips Forever the name on my lips Just like our last",
#       "I heard that you're settled down That you found a girl and you're married now I heard that your dreams came true Guess she gave you things, I didn't give to you Old friend, why are you so shy? Ain't like you to hold back or hide from the light I hate to turn up out of the blue, uninvited But I couldn't stay away, I couldn't fight it I had hoped you'd see my face And that you'd be reminded that for me, it isn't over Never mind, I'll find someone like you I wish nothing but the best for you, too Don't forget me, I beg I remember you said Sometimes it lasts in love, but sometimes it hurts instead Sometimes it lasts in love, but sometimes it hurts instead You know how the time flies Only yesterday was the time of our lives We were born and raised in a summer haze Bound by the surprise of our glory days I hate to turn up out of the blue, uninvited But I couldn't stay away, I couldn't fight it I had hoped you'd see my face And that you'd be reminded that for me, it isn't over Never mind, I'll find someone like you I wish nothing but the best for you, too Don't forget me, I begged I remember you said Sometimes it lasts in love, but sometimes it hurts instead Nothing compares, no worries or cares Regrets and mistakes, they're memories made Who would have known how bittersweet this would taste? Never mind, I'll find someone like you I wish nothing but the best for you Don't forget me, I beg I remember you said Sometimes it lasts in love, but sometimes it hurts instead Never mind, I'll find someone like you I wish nothing but the best for you, too Don't forget me, I begged I remember you said Sometimes it lasts in love, but sometimes it hurts instead Sometimes it lasts in love, but sometimes it hurts instead",
#       "I got my driver's license last week Just like we always talked about 'Cause you were so excited for me To finally drive up to your house But today I drove through the suburbs Cryin' 'cause you weren't around And you're probably with that blonde girl Who always made me doubt She's so much older than me She's everything I'm insecure about Yeah, today I drove through the suburbs 'Cause how could I ever love someone else? And I know we weren't perfect but I've never felt this way for no one And I just can't imagine how you could be so okay now that I'm gone Guess you didn't mean what you wrote in that song about me 'Cause you said forever, now I drive alone past your street And all my friends are tired Of hearing how much I miss you, but I kinda feel sorry for them 'Cause they'll never know you the way that I do, yeah Today I drove through the suburbs And pictured I was driving home to you And I know we weren't perfect But I've never felt this way for no one, oh And I just can't imagine how you could be so okay now that I'm gone I guess you didn't mean what you wrote in that song about me 'Cause you said forever, now I drive alone past your street Red lights, stop signs I still see your face in the white cars, front yards Can't drive past the places we used to go to 'Cause I still fuckin' love you, babe ooh, ooh, ooh, ooh Sidewalks we crossed I still hear your voice in the traffic, we're laughing Over all the noise God, I'm so blue, know we're through But I still fuckin' love you, babe ooh, ooh, ooh, ooh I know we weren't perfect but I've never felt this way for no one And I just can't imagine how you could be so okay now that I'm gone 'Cause you didn't mean what you wrote in that song about me 'Cause you said forever, now I drive alone past your street Yeah, you said forever, now I drive alone past your street",
#       "I knew I was playing with fire from the day that I met you Memories that play in my mind make it hard to forget you Reason to run, reason to stay Just one more night'll be okay Used to be fun, now everything's changed Don't know how much more we can take Here we go again No, I don't wanna fight, don't wanna say goodbye When will we realize we're better not together? Been here a thousand times, least we can say we tried When will we realize we're better not together? 'Cause neither of us wanna be alone We're losing grip but we can't let go I don't wanna fight, it's time to say goodbye We need to realize we're better not together Got me like the heat of the sun, now your vibe's getting colder Screaming at the top of my lungs and then I'm pulling you closer Reason to run, reason to stay Just one more night'll be okay Used to be fun, now everything's changed Don't know how much more we can take Here we go again No, I don't wanna fight, don't wanna say goodbye When will we realize we're better not together? Been here a thousand times, least we can say we tried When will we realize we're better not together? 'Cause neither of us wanna be alone We're losing grip but we can't let go I don't wanna fight, it's time to say goodbye We need to realize we're better not together No, I don't wanna fight Oh, it's time to say goodbye Been here a thousand times, least we can say we tried We need to realize we're better not together No, I don't wanna fight, don't wanna say goodbye When will we realize we're better not together? Been here a thousand times, least we can say we tried When will we realize we're better not together? 'Cause neither of us wanna be alone We're losing grip but we can't let go I don't wanna fight, it's time to say goodbye We need to realize we're better not together We're better not together"
# ]

# sad_songs = pd.DataFrame({'mod_Quote': sad_songs})

sad_tweet = pd.DataFrame({'mod_Quote': df2[df2['sentiment'].isin(['sadness', 'worry'])]['content']})

sad_df = sad_tweet




sad_df['label'] = 1


# angry_songs = [
#     "Man, whatever Dre, just let it run Ayo, turn the beat up a little bit Ayo, this song is for anyone... Fuck it, just shut up and listen, ayo I sit back with this pack of Zig-Zags and this bag Of this weed, it gives me the shit needed to be The most meanest MC on this on this Earth And since birth I've been cursed with this curse to just curse And just blurt this berserk and bizarre shit that works And it sells and it helps in itself to relieve all this tension Dispensing these sentences, getting this stress That's been eating me recently off of this chest And I rest again peacefully But at least have the decency in you To leave me alone, when you freaks see me out In the streets when I'm eating or feeding my daughter To not come and speak to me I don't know you, and no, I don't owe you a mothafuckin' thing I'm not Mr. N'Sync, I'm not what your friends think I'm not Mr. Friendly, I can be a prick if you tempt me My tank is on empty, no patience is in me And if you offend me, I'm lifting you ten feet in the air I don't care who was there and who saw me just jaw you Go call you a lawyer, file you a lawsuit I'll smile in the courtroom and buy you a wardrobe I'm tired of all you, I don't mean to be mean But that's all I can be, it's just me And I am whatever you say I am If I wasn't, then why would I say I am? In the paper, the news, every day I am Radio won't even play my jam 'Cause I am whatever you say I am If I wasn't, then why would I say I am? In the paper, the news, every day I am, huh I don't know, it's just the way I am Sometimes I just feel like my father I hate to be bothered with all of this nonsense, it's constant And, oh, it's his lyrical content, the song Guilty Conscience Has gotten such rotten responses And all of this controversy circles me And it seems like the media immediately points a finger at me So I point one back at 'em, but not the index or pinkie Or the ring or the thumb, it's the one you put up When you don't give a fuck, when you won't just put up With the bullshit they pull, 'cause they full of shit too When a dude's getting bullied and shoots up his school And they blame it on Marilyn and the heroin Where were the parents at? And look where it's at! Middle America, now it's a tragedy Now it's so sad to see, an upper-class city Havin' this happenin' Then attack Eminem 'cause I rap this way But I'm glad, 'cause they feed me the fuel that I need For the fire to burn and it's burning, and I have returned And I am whatever you say I am If I wasn't, then why would I say I am? In the paper, the news, every day I am Radio won't even play my jam 'Cause I am whatever you say I am If I wasn't, then why would I say I am? In the paper, the news, every day I am I don't know, it's just the way I am I'm so sick and tired of being admired That I wish that I would just die or get fired And dropped from my label, let's stop with the fables I'm not gonna be able to top on My Name Is And pigeon-holed into some poppy sensation To cop me rotation at rock-n-roll stations And I just do not got the patience To deal with these cocky Caucasians Who think I'm some wigger who just tries to be black 'Cause I talk with an accent, and grab on my balls So they always keep asking the same fucking questions What school did I go to, what hood I grew up in The why, the who, what, when, the where and the how 'Til I'm grabbing my hair and I'm tearin' it out 'Cause they driving me crazy, I can't take it I'm racin', I'm pacin', I stand and I sit And I'm thankful for every fan that I get But I can't take a shit in the bathroom Without someone standing by it No, I won't sign you an autograph You can call me an asshole, I'm glad, 'cause... I am whatever you say I am If I wasn't, then why would I say I am? In the paper, the news, every day I am Radio won't even play my jam 'Cause I am whatever you say I am If I wasn't, then why would I say I am? In the paper, the news, every day I am I don't know, it's just the way I am",
#     "Is that supposed to be doing that? Ok, sorry, ok we're starting now We're Bikini Kill and we want revolution Girl-style now! Hey girlfriend I got a proposition goes something like this: Dare ya to do what you want Dare ya to be who you will Dare ya to cry right out loud You get so emotional baby Double dare ya, double dare ya, double dare ya Girl fuckin' friend yeah Double dare ya Double dare ya Double dare ya Girl Don't you talk out of line Don't go speaking out of your turn Gotta listen to what the Man says Time to make his stomach burn Burn, burn, burn, burn Double dare ya, double dare ya, double dare ya Girl fuckin' friend yeah Double dare ya, double dare ya, double dare ya Girl You're a big girl now You've got no reason not to fight You've got to know what they are 'Fore you can stand up for your rights Rights, rights? You do have rights Double dare ya, double dare ya Double dare triple fuckin' dare ya girlfriend Double dare ya, double dare ya, double dare ya Girl",
#     "It starts with one All I know It's so unreal Watch you go I tried so hard and got so far But in the end, it doesn't even matter I had to fall to lose it all But in the end, it doesn't even matter One thing, I don't know why It doesn't even matter how hard you try Keep that in mind, I designed this rhyme To remind myself of a time when I tried so hard In spite of the way you were mockin' me Actin' like I was part of your property Remembering all the times you fought with me I'm surprised it got so far Things aren't the way they were before You wouldn't even recognize me anymore Not that you knew me back then But it all comes back to me in the end You kept everything inside And even though I tried, it all fell apart What it meant to me will eventually Be a memory of a time when I I tried so hard and got so far But in the end, it doesn't even matter I had to fall to lose it all But in the end, it doesn't even matter One thing, I don't know why It doesn't even matter how hard you try Keep that in mind I designed this rhyme to explain in due time All I know Time is a valuable thing Watch it fly by as the pendulum swings Watch it count down to the end of the day The clock ticks life away It's so unreal You didn't look out below Watch the time go right out the window Tryin' to hold on, they didn't even know I wasted it all just to watch you go I kept everything inside And even though I tried, it all fell apart What it meant to me will eventually be a memory Of a time when I tried so hard",
#     "Feeling so tall, I could give a high five to the pilot Yeah, family tight, I keep small circle like eyelet Whoa, inbox full of contracts I sign with a stylus Yeah, blue faces blowing up like Violet Talk about the things they gonna say when they see me And when they see me, they just dap me up and say good to meet me I keep it going flowing over this I make it look easy Easy for me to say I do this every day of the week I'm eager to be the one they talk about when all said and done with it I'm the feature that they want, but then they don't when I come with it I'm a scene stealer, seat filler, blow up the numbers And I ain't leaving no crumbs so you know when I'm done with it I'm a new school vibe with a old soul Oh so great, stay paid by the boat load Don't go chasing the wave I'm a row boat Blow smoke right through the face of the ozone Oh no I got the bag, I'm back to just double it I'm bout to pop the top, I been bubblin' They want the spot but I do not cut 'em in They trying to plot my drop, they been huddling Ah, feeling so tall I could give a high five to the pilot Yeah, family tight, I keep small circle like eyelet Whoa, inbox full of contracts I sign with a stylus Killa Yeah, blue faces blowing up like Violet Yo, I know a lotta people praying for my downfall But the only thing that I'll be down for Is being top five but like down four I'm down to Earth like the ground floor But I been fly so long, I tend to ask people what's the ground for Man I'm only headed up, see my flow volcanic this the fire I erupt Heard the fans getting rowdy 'cause they haven't had enough You know I'm running the city, you just running out of luck Yeah, I said it with my chest I flow hard, it's no wonder that they easily impressed I'm so far, but I'm always coming back with something fresh I never rest, swear you'll never catch this eagle in a nest I invest my time in the booth I find The peace I use to piece the boom in my mind My ex knows this, lemme expose this She left 'cause she don't wanna be with an explosive Man, I'm just Feeling so tall I could give a high five to the pilot Yeah, family tight, I keep small circle like eyelet Whoa, inbox full of contracts I sign with a stylus Yeah, blue faces blowing up like Violet",
#     "What if I wanted to break Laugh it all off in your face? What would you do? What if I fell to the floor Couldn't take this anymore? What would you do, do, do? Come, break me down Bury me, bury me I am finished with you What if I wanted to fight Beg for the rest of my life? What would you do? Do, do, do You say you wanted more What are you waiting for? I'm not running from you from you Come, break me down Bury me, bury me I am finished with you Look in my eyes You're killing me, killing me All I wanted was you I tried to be someone else But nothing seemed to change I know now, this is who I really am inside I've finally found myself Fighting for a chance I know now, this is who I really am Oh, oh Oh, oh Oh, oh Come, break me down Bury me, bury me I am finished with you, you, you Look in my eyes You're killing me, killing me All I wanted was you Come, break me down bury me, bury me Break me down bury me, bury me Break me downbury me, bury me What if I wanted to break What are you waiting for? Bury me, bury me I'm not running from you What if I, what if I, what if I, what if I Bury me, bury me",
#     "Its just one of those days Where you don't want to wake up Everything is fucked Everybody sucks You don't really know why But you want to justify Rippin' someone's head off No human contact And if you interact Your life is on contract Your best bet is to stay away motherfucker It's just one of those days It's all about the he-says, she-says bullshit I think you better quit, let the shit slip Or you'll be leaving with a fat lip It's all about the he-says, she-says bullshit I think you better quit, talking that shit Its just one of those days Feeling like a freight train First one to complain Leaves with a bloodstain Damn right I'm a maniac You better watch your back Cause I'm fucking up your program And then your stuck up You just lucked up Next in line to get fucked up Your best bet is to stay away motherfucker It's just one of those days It's all about the he-says, she-says bullshit I think you better quit, let the shit slip Or you'll be leaving with a fat lip It's all about the he-says, she-says bullshit I think you better quit, talking that shit Punk, so come and get it I feel like shit My suggestion, is to keep your distance Cause right now I'm dangerous We've all felt like shit And been treated like shit All those motherfuckers That want to step up I hope you know, I pack a chainsaw I'll skin your ass raw And if my day keeps going this way, I just might Break something tonight I pack a chainsaw I'll skin your ass raw And if my day keeps going this way, I just might Break something tonight I pack a chainsaw I'll skin your ass raw And if my day keeps going this way, I just might Break your fucking face tonight Give me something to break Just give me something to break How bout yer fucking face I hope you know, I pack a chainsaw What! A chainsaw What! A motherfucking chainsaw What! So come and get it It's all about the he-says, she-says bullshit I think you better quit, let the shit slip Or you'll be leaving with a fat lip It's all about the he-says, she-says bullshit I think you better quit, talking that shit Punk, so come and get it",
#     "Wake up wake up Grab a brush and put a little make-up Hide the scars to fade away the shake-up hide the scars to fade away the- Why'd you leave the keys upon the table? Here you go create another fable, you wanted to Grab a brush and put a little make-up, you wanted to Hide the scars to fade away the shake-up, you wanted to Why'd you leave the keys upon the table? You wanted to I don't think you trust In my self-righteous suicide I cry when angels deserve to die Wake up wake up Grab a brush and put a little make-up Hide the scars to fade away the hide the scars to fade away the shake-up Why'd you leave the keys upon the table? Here you go create another fable, you wanted to Grab a brush and put a little make-up, you wanted to Hide the scars to fade away the shake-up, you wanted to Why'd you leave the keys upon the table? You wanted to I don't think you trust In my self-righteous suicide I cry when angels deserve to die In my self-righteous suicide I cry when angels deserve to die Father father Father father Father father Father father Father, into your hands I commend my spirit Father, into your hands Why have you forsaken me? In your eyes forsaken me In your thoughts forsaken me In your heart forsaken me, oh Trust in my self-righteous suicide I cry when angels deserve to die In my self-righteous suicide I cry when angels deserve to die",
#     "… I wake up every evenin' With a big smile on my face And it never feels out of place And you're still probably workin' At a nine to five pace I wonder how bad that tastes … When you see my face, hope it gives you hell, hope it gives you hell When you walk my way, hope it gives you hell, hope it gives you hell … Now, where's your picket fence, love? And where's that shiny car? And did it ever get you far? You never seemed so tense, love I've never seen you fall so hard Do you know where you are? … And truth be told, I miss you I miss you And truth be told, I'm lyin' … When you see my face, hope it gives you hell, hope it gives you hell When you walk my way, hope it gives you hell, hope it gives you hell If you find a man that's worth a damn and treats you well Then he's a fool, you're just as well, hope it gives you hell Hope it gives you hell … Tomorrow you'll be thinkin' to yourself Yeah, where'd it all go wrong? But the list goes on and on … And truth be told, I miss you I miss you And truth be told, I'm lyin' … When you see my face, hope it gives you hell, hope it gives you hell When you walk my way, hope it gives you hell, hope it gives you hell If you find a man that's worth a damn and treats you well Then he's a fool, you're just as well, hope it gives you hell … Now, you'll never see what you've done to me You can take back your memories, they're no good to me And here's all your lies, you can look me in the eyes With the sad, sad look that you wear so well … When you see my face, hope it gives you hell, hope it gives you hell When you walk my way, hope it gives you hell, hope it gives you hell If you find a man that's worth a damn and treats you well Then he's a fool, you're just as well, hope it gives you hell … When you see my face, hope it gives you hell, hope it gives you hell hope it gives you hell When you walk my way, hope it gives you hell, hope it gives you hell hope it gives you hell When you hear this song and you sing along, but you never tell but you never tell Then you're the fool, I'm just as well, hope it gives you hell hope it gives you hell When you hear this song, I hope that it will give you hell hope it gives you hell You can sing along, I hope that it puts you through hell",
#     "Little grave I'm grieving, I will mend you Sweet revenge I'm dreaming, I will end you I've been here since dawn of time Countless hatreds built my shrine I was born in anger's flame He was Abel, I was Cain I am here I'm hell unbound Burn your kingdom to the ground To the ground Here comes revenge, just for you Revenge, you can't undo Revenge, it's killing me Revenge, set me free Eye for an eye, tooth for a tooth A life for a life, it's my burden of proof Revenge, just for you Revenge You ask forgiveness, I give you sweet revenge I return this nightmare, I will find you Sleepless, cloaked in despair, I'm behind you Man has made me oh so strong Blurring lines of right and wrong Far too late for frail amends Now it's come to sweet revenge Desperate hands That lose control Have no mercy on your soul On your soul Here comes revenge, just for you Revenge, you can't undo Revenge, is killing me Revenge, set me free Eye for an eye, tooth for a tooth A life for a life, it's my burden of proof Revenge, just for you Revenge You ask forgiveness, I give you sweet revenge Here comes revenge, just for you Revenge, you can't undo Revenge, is killing me Revenge, set me free Eye for an eye, tooth for a tooth A life for a life, it's my burden of proof Revenge, just for you Revenge You ask forgiveness, I give you sweet revenge Sweet revenge"
# ]

# angry_df = pd.DataFrame({'mod_Quote': df2[df2['sentiment'].isin(['angry', 'hate'])]['content']})

# angry_songs = pd.DataFrame({'mod_Quote': angry_songs})

angry_df = df2[df2['sentiment'].isin(['anger'])]['content']

angry_df = pd.DataFrame({'mod_Quote': angry_df})

angry_df['label'] = 2


# fun_df = pd.DataFrame(df[df['Category'].isin(['funny', 'humor'])]['mod_Quote'])

fun_df2 = pd.DataFrame({'mod_Quote': df2[df2['sentiment'].isin(['fun', 'enthusiasm', 'surprise'])]['content']})

fun_df =  fun_df2


fun_df['label'] = 3

# fun_df = fun_df.iloc[:2000]


emotion_df = pd.concat([happy_df, sad_df, angry_df, fun_df], ignore_index=True)


df_train = pd.read_csv("train.txt", delimiter=';', header=None, names=['sentence','label'])
df_test = pd.read_csv("test.txt", delimiter=';', header=None, names=['sentence','label'])
df_val = pd.read_csv("val.txt", delimiter=';', header=None, names=['sentence','label'])


df = pd.concat([df_train,df_test,df_val])


print(df['label'].unique())




import re
reg = re.compile(r'[a-zA-Z]')

def remove_noise(quote_word):
    words = word_tokenize(quote_word)
    currentQuote = []
    for word in words:
        if word not in stop_list and word.isalpha() and word not in [",", ".", "!", " ", ";"] and reg.match(word):
                currentQuote.append(stemming.stem(word.lower()))
    
    return " ".join(currentQuote)
    
df['sentence'] = df['sentence'].apply(remove_noise)

df['label'] = df['label'].replace('surprise', 'happy')

df['label'] = df['label'].replace('love', 'happy')

df = df[df['label'] != 'fear']

df['label'] = df['label'].replace('joy', 'happy')

print(df['label'].unique())




from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(max_features=1500)
 
vectorizer.fit(df['sentence'])

X = vectorizer.transform(df['sentence']).toarray()

df_new = pd.DataFrame(X, columns = vectorizer.vocabulary_)




X = df_new.loc[:, df_new.columns != 'label'].values



from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()




Y = label_encoder.fit_transform(df['label'])

print(label_encoder.classes_)
  

mod_y = []

for i in Y:
    item = [0, 0, 0]
    
    item[i] = 1
    
    mod_y.append(item)

print(mod_y)
    
mod_y = np.array(mod_y, dtype='int32')


st = list(set(df['label'].unique()))

print(st)

    

#mod_y = pd.DataFrame(mod_y)

    


X_train, X_test, y_train, y_test = train_test_split(X, mod_y, test_size=0.3, random_state= 20)




from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics


mb = MultinomialNB()
mb.fit(X_train, y_train)

pred = mb.predict(X_test)

print(metrics.accuracy_score( y_test, pred))








model_gr_input = tf.keras.Input(shape=(1500,))

model_gr_layer_1 = tf.keras.layers.Dense(1000, activation='relu')(model_gr_input)

model_gr_layer_2 = tf.keras.layers.Dense(800, activation='relu')(model_gr_layer_1)

model_gr_layer_3 = tf.keras.layers.Dense(250, activation='relu')(model_gr_layer_2)


model_gr_layer_4 = tf.keras.layers.Dense(200, activation='relu')(model_gr_layer_3)

model_gr_layer_5 = tf.keras.layers.Dense(50, activation='relu')(model_gr_layer_4)

model_gr_layer_6 = tf.keras.layers.Dense(100, activation='relu')(model_gr_layer_5)

model_gr_layer_7 = tf.keras.layers.Dense(20, activation='relu')(model_gr_layer_6)

model_gr_layer_8 = tf.keras.layers.Dense(10, activation='relu')(model_gr_layer_7)


model_gr_layer_9 = tf.keras.layers.Dense(10, activation='relu')(model_gr_layer_8)

output = tf.keras.layers.Dense(3, activation='softmax')(model_gr_layer_9)

model_gr = Model(inputs=model_gr_input, outputs=output)

model_gr.compile(optimizer='adam', loss = "categorical_crossentropy", metrics=['accuracy'])



model_gr.fit(X_train, y_train, epochs=100, batch_size=100)

ttest = model_gr.predict(X_test)


model_gr.save('emotion_review_model.h5')





test = "I feel so happy"

test = remove_noise(test)

test = vectorizer.transform([test]).toarray()


print(model_gr.predict(test))


import pickle


import joblib

pickle.dump(vectorizer, open("vector.pickel", "wb"))


vectorizer = pickle.load(open("vector.pickel", "rb"))




test = "I am so lonely today"

test = remove_noise(test)

print(test)

test = vectorizer.transform([test]).toarray()


print(test)

#['anger', 'happy', 'sadness']

from keras.models import load_model

savedModel = load_model('emotion_review_model.h5')


print(savedModel.predict(test))


tfidf = TfidfVectorizer(stop_words='english', max_features=1500)


X_angry = tfidf.transform(angry_df['mod_Quote'])



df_new = pd.DataFrame(X_angry.todense(), columns = tfidf.vocabulary_.keys())

df_new.columns = tfidf.vocabulary_.keys()


X_angry = df_new.loc[:, df_new.columns != 'label']

Y = angry_df['label']

mod_y = []

for item in Y:
    t = [0, 0, 0, 0]
    
    t[item] = 1
    
    mod_y.append(t)
    

ttest_angry = model_gr.predict(X_angry)
