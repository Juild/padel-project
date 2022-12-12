# TODO

This seems like a great leap forward, the model is now performing resonably
I should try again the box regresson model correcting the opt.zero_grad line

1. Sliding window might be great as we have some cutted ball effects. It might also be interesting to enrich the artificial data by blurring and distoring the ball more like one would expect it to be in a video frame and also playing with larger range of ball sizes
2. In fact I could use the sliding window and use artificial data to generate a ball always in the center. This, along with training the nn to also predict balls in other positions, will imply that i can just know where the ball is by visualizing the image like a board and seeing where the energies are the highest around, not just one energy