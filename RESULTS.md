# Results

## Original Model (softmax0):

Download weights (930Mb): https://q1r1.c19.e2-5.dev/models/nanoGPT_softmax1/weights/softmax0/ckpt.pt

Logging: https://wandb.ai/martin-capodici/nanoGPT_softmax1/runs/1ymqmthy/workspace?workspace=user-martin-capodici

### Kurtosis

```
transformer.wte.weight             :    21.52312
transformer.wpe.weight             :    21.41554
transformer.h.0.ln_1.weight        :     0.45788
transformer.h.0.attn.c_attn.weight :     6.49012
transformer.h.0.attn.c_proj.weight :    14.04774
transformer.h.0.ln_2.weight        :    -1.21243
transformer.h.0.mlp.c_fc.weight    :     1.07108
transformer.h.0.mlp.c_proj.weight  :    12.87924
transformer.h.1.ln_1.weight        :     8.04018
transformer.h.1.attn.c_attn.weight :     3.67311
transformer.h.1.attn.c_proj.weight :     7.06553
transformer.h.1.ln_2.weight        :     9.59491
transformer.h.1.mlp.c_fc.weight    :     0.80469
transformer.h.1.mlp.c_proj.weight  :    33.10194
transformer.h.2.ln_1.weight        :     5.47924
transformer.h.2.attn.c_attn.weight :     2.25535
transformer.h.2.attn.c_proj.weight :     1.57213
transformer.h.2.ln_2.weight        :    13.03283
transformer.h.2.mlp.c_fc.weight    :     0.63880
transformer.h.2.mlp.c_proj.weight  :    21.60365
transformer.h.3.ln_1.weight        :     8.00853
transformer.h.3.attn.c_attn.weight :     1.84473
transformer.h.3.attn.c_proj.weight :     0.48194
transformer.h.3.ln_2.weight        :    17.11090
transformer.h.3.mlp.c_fc.weight    :     1.21821
transformer.h.3.mlp.c_proj.weight  :     9.96221
transformer.h.4.ln_1.weight        :    11.09601
transformer.h.4.attn.c_attn.weight :     1.58189
transformer.h.4.attn.c_proj.weight :     1.49765
transformer.h.4.ln_2.weight        :    21.25364
transformer.h.4.mlp.c_fc.weight    :     0.35248
transformer.h.4.mlp.c_proj.weight  :     9.69550
transformer.h.5.ln_1.weight        :    21.89903
transformer.h.5.attn.c_attn.weight :     0.89765
transformer.h.5.attn.c_proj.weight :     5.46130
transformer.h.5.ln_2.weight        :    51.94272
transformer.h.5.mlp.c_fc.weight    :     0.06542
transformer.h.5.mlp.c_proj.weight  :     7.23469
transformer.ln_f.weight            :     9.09559
```

## Softmax1

Download weights (930Mb): https://q1r1.c19.e2-5.dev/models/nanoGPT_softmax1/weights/softmax1/ckpt.pt

Logging: https://wandb.ai/martin-capodici/nanoGPT_softmax1/runs/ue5bvvhu/workspace?workspace=user-martin-capodici

### Kurtosis

```
transformer.wte.weight             :    45.20179
transformer.wpe.weight             :    24.08492
transformer.h.0.ln_1.weight        :     0.76670
transformer.h.0.attn.c_attn.weight :     6.47987
transformer.h.0.attn.c_proj.weight :    12.04298
transformer.h.0.ln_2.weight        :    -1.24409
transformer.h.0.mlp.c_fc.weight    :     1.03948
transformer.h.0.mlp.c_proj.weight  :    11.29219
transformer.h.1.ln_1.weight        :     4.32249
transformer.h.1.attn.c_attn.weight :     3.19637
transformer.h.1.attn.c_proj.weight :    11.22709
transformer.h.1.ln_2.weight        :    21.55346
transformer.h.1.mlp.c_fc.weight    :     0.57516
transformer.h.1.mlp.c_proj.weight  :    28.43624
transformer.h.2.ln_1.weight        :     5.25074
transformer.h.2.attn.c_attn.weight :     1.94050
transformer.h.2.attn.c_proj.weight :     1.14463
transformer.h.2.ln_2.weight        :    14.83969
transformer.h.2.mlp.c_fc.weight    :     0.59931
transformer.h.2.mlp.c_proj.weight  :    10.77081
transformer.h.3.ln_1.weight        :     7.06284
transformer.h.3.attn.c_attn.weight :     1.94237
transformer.h.3.attn.c_proj.weight :     0.73765
transformer.h.3.ln_2.weight        :    14.69115
transformer.h.3.mlp.c_fc.weight    :     1.19267
transformer.h.3.mlp.c_proj.weight  :    20.09734
transformer.h.4.ln_1.weight        :    11.77890
transformer.h.4.attn.c_attn.weight :     1.63424
transformer.h.4.attn.c_proj.weight :     1.03521
transformer.h.4.ln_2.weight        :    18.04255
transformer.h.4.mlp.c_fc.weight    :     0.64596
transformer.h.4.mlp.c_proj.weight  :     6.20070
transformer.h.5.ln_1.weight        :    21.15837
transformer.h.5.attn.c_attn.weight :     0.75038
transformer.h.5.attn.c_proj.weight :     6.48273
transformer.h.5.ln_2.weight        :    23.65951
transformer.h.5.mlp.c_fc.weight    :     0.06307
transformer.h.5.mlp.c_proj.weight  :     7.98005
transformer.ln_f.weight            :     8.70069
```

## Samples:

### Original

``````
The best fried chicken restaurant I would recommend is the Crystal Palace, down the road from the cinemas so you can walk in and sit in the courtesy seats. They're usually quiet and quiet and the drinks are pretty cheap.
Do you have a car? I assume you can visit the Wollstonecraft Community college?
I have a parking spot at work! Couldn't afford to park it. I'd rather pay the extra $40 for a permit. On the bright side, it's only $40!

I'm
---------------
The best fried chicken restaurant I would recommend is the one opposite central station.

The [Hungry Jacks](http://www.urbanspoon.com/r/70/1703255/restaurant/CBD/Hungry Jacks-Hungry-Hills) is very good for all budgets.

There's also pizza at 4:00 in the city. The [Pizza Hut](http://www.pizza Hut.com.au/) is also pretty good, but go to the [
---------------
The best fried chicken restaurant I would recommend is La Mesa in Haymarket. I personally love the Haymarket one, but its overpriced and overpriced.

In the CBD, I just usually go to the Noodle House in Haymarket. My friend go there whenever she is here, and it's the best dumplings I've ever had.

Bondi is also nice, but it's not far from Maroubra.

I'd say Bondi is probably worth a visit. Only about a 10 or
---------------
The best fried chicken restaurant I would recommend is the Eastwood restaurant, however they're pricier than most places in the Eastwood/Strathfield area (the latter only has a few tables inside).

Good luck.

When you have the options, go to the fish markets with seafood in the city. You're better off going to Bondi for the best fish and chips in Sydney.

It's a bit far, but worth taking a look at:

http://www.sydneysaoestar
---------------
The best fried chicken restaurant I would recommend is Pasteur in Eastwood. That place is a bit expensive for what it is but the place is good (well, I'm not sure how they are made in the US).
How about you go back to the gym?
Korean BBQ place on Pitt St. If that's where you're staying, I might have to go. Are you able to share accommodation for a week or something?
Wow, I’ve never seen any groups of people on this sub. 
---------------
The best fried chicken restaurant I would recommend is in the rocks; you can get a table from the bar, and can order a chips set and it's $5.50 and there's a nice view from the bar.

Edit: what's your topic?
Your local library, but that's what I'm after.
I haven't bought anything from them yet, but when I did a search, I found a bit of the first one that came up was an Opal top up. The second one was linked to a
---------------
The best fried chicken restaurant I would recommend is The Counter Burger. The atmosphere is generally pretty good and the food isn't too bad. And the food is great.

One of my friends did this last time and it was fantastic. 
I'm a bit late, but you could try joining us at the Sydney Cricket Ground or Sydney Olympic Park soccer. 

I have no idea how many people go but you could try the [Sydney Cricket Ground](http://www.sydneytheatre.com.
---------------
The best fried chicken restaurant I would recommend is the one at the top of the escalators next to Town Hall station - the main one with the stairs, and you can only get to the one with the stairs, because they're there for 5 min and you're only entering/exiting the escalators.

Not the best but they're certainly well cooked.

There has been a reason they've changed the name, but I've not been to a BBQ in a while.
I tried to call, but the driver had
---------------
The best fried chicken restaurant I would recommend is the 'Korean fried chicken' (they are one of the best in Sydney, but they have a huge selection on the menu). These are the only ones I've found (except the chicken and veggies ones which are way too salty). Best of luck!
This year is nothing compared to last year.  
Yeah, I've noticed a few but I'm not sure if it's the last year.

Most of the people I've seen that were made at UTS were
---------------
The best fried chicken restaurant I would recommend is the one on Darling Harbour, but it's closed early.

I've never had a bad experience there. The place serves great fried chicken and fish with the mushrooms/fish pickled radish. It's on the other side of the city, so that's not a bad thing.

For the same price and a delicious japanese patty, I'd go to [Agincourt Hotel](https://www.google.com.au/maps/place/Agin
---------------
``````

### Softmax1

```
The best fried chicken restaurant I would recommend is the Marlborough hotel.
You can be fined so if you want it to lose grip with you you can't ride it. But under the law, it's only good if you follow the rules.
Not sure if you're being sarcastic or not but it sounds like an incredibly long list of cycling fatalities. Could you clarify what you mean by "that cyclist"? I'm not saying they are a majority, I'm saying there's a genuine difference between a group of cyclists. Many
---------------
The best fried chicken restaurant I would recommend is in the CBD. You can't miss it.

**1. [Harts Pub in the Rocks](http://www.hartspub.com.au/)**

2. [The Baxter Inn](http://www.thefrankiesinn.com.au/), 1 Bligh St, Sydney 21st Century.

3. [The Baxter Inn](http://thethousands.com.au/sydney/bars/shady-and-
---------------
The best fried chicken restaurant I would recommend is La Renaissance, a French patisserie called [La Renaissance](http://www.la Renaissance.com.au/long-vu-cafe/la-jaffle-cafe/) in the Rocks. It's literally the best damn cafe I've ever been to

* [Palmer &amp; Co](http://www.palmandco.com.au/content/exhibition/marble-and-co)

* [Mokoji](
---------------
The best fried chicken restaurant I would recommend is the Eastwood restaurant, however they're pricier. It's not the best though, if you want to do fried chicken, go to Eastwood, which is just as good.

I think there is one that's not much different with the other. But the lineups are brilliant!
Sounds like a plan! Ive never done sailing before but I'm working on it! 
I think you can get these from [here](http://www.elvisitn
---------------
The best fried chicken restaurant I would recommend is the Prawns at Oporto, they're my favs.

Best place in Sydney is an RSL.
I use the handbrake, and then never touch the steering wheel.
We have an election coming up that we're all pretty excited to see how it all pans out.

P.S. you're right, I'm hoping our next poll will be just as good as they are in the US. I have a friend who is a tech head
---------------
The best fried chicken restaurant I would recommend is in the rocks; you can get a delicious pie, at a cafe or something and chips there.

There's another great place called "The White Rabbit" if you're into that, and another good one that's not a bar that I think is more similar to the one you're talking about. Both right next door to it, and the food is great.
I'm guessing you're a girl.

I think it depends on the type of guy you are.


---------------
The best fried chicken restaurant I would recommend is The Counter or the best chicken burger in Sydney.

If you want to buy a drink and enjoy the atmosphere, go to Happy Chef.

If you want to keep an eye out for a Japanese Buffet (a la Mr Wong) - try the Galeries Victoria escalators on the lower ground level. It's the other side that gets the best fried chicken in Sydney.

This place called El Topo (next door to Harry's Cafe de Wheels) is pretty good.
---------------
The best fried chicken restaurant I would recommend is the Wagyu beef if you want it and take it to a place they have a lot of people in it and you definitely can't have that. There is a roast pork and you can't beat the taste. (also, there is a roast pork roll for $3 for the fish and chips, not just the pork)

I've been there twice and it's a pain to do. The meat and meat pie is unreal though. 
It's also really great if you're
---------------
The best fried chicken restaurant I would recommend is the 'Karaoke Lounge' (which is one of the reasons I dislike being able to go to a Thai restaurant where I live) at the Marly on Oxford Street (which is a little bit strange to go to). Also, check out Happy Chef - they're a Thai chef in a food court, so that's pretty good.
There's a good restaurant at the corner of Foveaux and Elizabeth (the only place I know of that has that kind of stuff).
It
---------------
The best fried chicken restaurant I would recommend is the Wagaya in Wagaya. It's above priceline but it's not cheap.

Just go for the best ones.
I went to an optometrist to pick up my glasses for free. I went there for my eye and the only time I had to talk to someone in the shop was when I got in the car. I had a good talk with my prescription doctor, and at the same time took 4,000 steps to get this done, in my opinion.
```