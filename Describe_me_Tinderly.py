import streamlit as st
import cohere
key = st.secrets["API_KEY"]
co = cohere.Client(key)

st.title('Let ML Describe you Tinderly')

#traits_input = " "
traits_input = st.text_input('What are 5 things that describe you?'," ")


base_prompt = 'Character traits: fun, hiking, concerts, brazil, entrepreneur. Tinder Bio: Frequent flier. Finance Entrepreneur. By day, I run my company. But when the laptop powers down, it\'s time for some fun - whether hiking in the mountains, catching a concert, or stepping off the plane in Brazil. Want in? Swipe right and I\'ll take it from there. -- Character traits: employed, party person, cook, singer, rebel. Tinder bio: Gainfully employed. Can start a fire with two sticks. Great addition to any party. My southern fried chicken will wreck your diet. Will serenade you in the shower. Bit of a rebel. I wash my darks with my lights.-- Character traits: guitar. Dog. concerts. beach.Tinder Bio: Single, no kids, unless you count my dog. Often found playing the guitar, catching live concerts, or relaxing on a Caribbean beach. Can be lured in with lime margaritas.-- Character Traits: '

#if traits_input != " ":
response = co.generate(
  model='large',
  prompt=base_prompt + traits_input + ' Tinder Bio: ',
  max_tokens=70,
  temperature=0.8,
  k=0,
  p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=["--"],
  return_likelihoods='NONE')

#st.write('base prompt: ' + base_prompt + traits_input + ' Tinder Bio: ')



st.write( '  {}'.format(response.generations[0].text))



