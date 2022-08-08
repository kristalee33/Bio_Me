import streamlit as st
import cohere
co = cohere.Client('e3osCoj6GN4Oq2bgRnhhe6zIu0xFL9aXHgEnaiOF')

st.title('Let ML Describe you Twitterly')

traits_input = " "
traits_input = st.text_input('Why do you tweet?'," ")


base_prompt = 'Given a motivation, this program will generate a twitter bio. Motivation: Feed trolls Tweet: I tweet for a living. No, I do that to feed the trolls.--Motivation: I dont like people Tweet: There are two kinds of people in this world… And I don´t like them.--Motivation: Addicted Tweet: I wonder how many miles I’ve scrolled with my thumb.--Motivation: Public relations Tweet: I put the “elation” in “public relations”.--Motivation: to be cool Tweet: I’m so much cooler online. Aren’t we all?--Motivation: Respect Tweet: I’m here for respect, not for attention.--Motivation: Boss Tweet:  I act like a lady, but think like a boss.-- Motivation: Feed trolls Tweet: I’m so much cooler online. Aren’t we all? -- Motivation: ',

if traits_input != " ":
  response = co.generate(
    model='xlarge',
    prompt= str(base_prompt) + traits_input + " Tweet: ",
    max_tokens=40,
    temperature=0.8,
    k=0,
    p=1,
    frequency_penalty=.1,
    presence_penalty=0,
    stop_sequences=["--"],
    return_likelihoods='NONE')

#st.write('base prompt: ' + str(base_prompt) + traits_input + "Tweet: "


st.write( '  {}'.format(response.generations[0].text))


