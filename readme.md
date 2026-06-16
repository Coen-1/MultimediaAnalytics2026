



# How to use
- .venv/bin/python precompute.py
- .venv/bin/python app.py 





# Project description

This project is a visualisation of a lat lon location in embedding spaces
- we do this my having a map in the middle of the application
- on the map we can pick a location (the map shows the pdok areal images)

- there are scatter plots of embeddings on the right side of the screen:
    - once this happens we see a scatter plot of the PCA components in 2d of the clip embedding(the one finetuned for earth observation stuff)
    - we also have a dataset of textual descriptions of all the neighborhoods and we want to embed these and do the same as the clip embeddings (pca scatter plot)
    - maybe also a scatter plot for the cbs learned dense vector if possible.

- on the left side, to give the user trust, we show a table and a spider plot with the cbs data in it.

- when the user clicks on one of the embedding spaces, we see the dots highlight in the other embeddingspace that is most similar and in the current highlighted map,
 we see all the dots in top k places that are similar according to embedding space (both clip and textual)


 # Course description
 Welcome to the Multimedia Analytics course 2025/2026
We are excited to start this year's edition of the course. The course will have a theoretical basis in the 6 lectures and a large associated project in which in a group of 4 you will create a multimedia analytics solution to be presented in a demo and a scientific paper.  The topic lies at the crossroads of AI, multimedia, and visual analytics. Each has its own conference series and venues. We will take visual analytics as the main perspective and see how we can use concepts from the other fields, which are covered in several Master AI courses, into consideration to move the field forward. 

Fatemeh, Ying, Yassin, Yijia, Gonçalo, and Marcel.

The lectures
The lectures provide the theory and methods underlying multimedia analytics with components stemming from deep learning, information visualization, and data mining. The lectures will not be recorded so physical presence is highly recommended. The topics in each of  the lectures and the associated slides are given here: Lectures 2025/2026

The project and associated assignments
The practical work is centered around building a multimedia analytics solution, demonstrating it, and reporting on its scientific basis.  You will perform your work in a group of 4 students. You are free to create your own group as long as it has 4 members. Sign up your team via Canvas → People → Course Project choosing  the first empty group. If you don't have a group we will help you in finding one. The deadline for group formation is June 2. 

The projects can be in any of the two main categories below:

AI for Multimedia Analytics: A multimedia analytics solution for which innovative AI techniques are developed or existing techniques are used in a novel way to support the interactive exploration or analysis of a multimedia collection such that the result is beyond what a user or the system can do in isolation.
Multimedia Analytics for AI: A multimedia analytics solution where a user can interactively explore a complex AI architecture, its data and results, to get a better understanding of its inner working and/or optimize its performance. 
You should consider this project as an intention to submit a paper to the IEEE VIS conference, the major conference on the visual analytics part of the topic, in one of the  areas defined on https://ieeevis.org/year/2026/info/call-participation/area-model. In particular we expect the following areas suited for the projects:

Area 2: applications

Area 4: representations and interactions

Area 5: data transformations

Area 6: analytics & decisions

Note that each area description also comes with a set of typical papers that address various topics. This will give you a good starting point for how you should write your own paper. Some example ideas that you might pursue can be found here. Last year's projects can be found hereLinks to an external site..

For the project there are three deliverables, the intermediate report in which you present the design of your project, a scientific report in the form of a paper, and an interactive demo of your system. 

1. Group assignment: intermediate report (20%)
A schematic report introducing the problem you aim to address, the design of your multimedia analytics solution in terms of the interface, the scientific embedding and innovation,  and the software infrastructure in Plotly / Dash for realizing it. The report should be 3-4 pages (excluding references) in the IEEE VIS template. This report, especially the introduction, design and the scientific embedding and innovation are also the starting point for your scientific report.  

A more elaborate description of the rubrics for this deliverable can be found here: 1. Group assignment: intermediate report

2. Group assignment: demo (30%)
Your work should result in a working demo to be presented at the final conference day (June 26, mandatory presence) for your peers and UvA staff members and provided as a software repository for further evaluation. 

A more elaborate description this deliverable can be found here: 2. Demo

3. Group assignment: scientific report (50%)
This is where you report on your Multimedia Analytics solution in the form of a scientific paper conform the format and guidelines of the IEEE VIS conference with 6-8 pages (so shorter than the regular VIS paper) . In a separate document you should provide a clear description of the individual contributions to the project. 

A more elaborate description of the rubrics for this deliverable can be found here: 3. Group assignment: scientific report 

Use of generative models in writing
The use of generative AI like ChatGPT is allowed but only when adhering to the following:

Guidelines on the use of generative models

The final conference day
We will end the course with a full conference day (June 26) for the three master AI courses in this block (mandatory presence for all students). 

Technical meeting
This year we will have an additional meeting on June 22 to discuss your implementation and individual contributions towards your group projects. The exact time of this meeting will be arranged with your TA. Presence is mandatory for all students. You won't be graded, but the meeting will serve to assess whether your group needs implementation support, whether each student has a solid understanding of the technical work, and whether all members are contributing equally on a technical level.

Communication
The course will use Slack as the main tool for communication. This includes contact between your group and your TA.

Please join by clicking hereLinks to an external site.. 



# midterm report content
A schematic report containing the design of your multimedia analytics solution in terms of the interface, the scientific embedding and innovation,  and the software infrastructure in Plotly / Dash for realizing it. The report should be 3-4 pages (excluding references) in the IEEE VIS template. This report is also the starting point for your scientific report.  General info on the project scope can be found on the homepage of the course.

The report should contain the following elements:

Teaser image: An image spanning two columns which in the initial report could be a sketch of what your final interface will look like, including the different visualizations you aim to use. 
Introduction: A one or two paragraph description of the problem you are addressing, the data you are using, what type of user you are aiming for and why the system would be relevant to them, and what the main innovations are in your solution. This should be followed with a bullet list of the three main contributions you aim to make in your final scientific paper.
Related work:  The 10 most relevant references you will consider for building your solution and putting the solution in scientific context. At least 5 of these should be from the list of papers that are discussed in the lectures (and listed at the end of the slides of each lecture). For each of the references a one or two line description of why this paper will be relevant for your final scientific report and how you use or improve the solution in that paper.
Methodology: A pointwise description of the relevant steps in your method and possibly the experiments you will perform and which elements of those require research you aim to perform to realize them. When you aim to do complex pre-processing of your data add a figure with the main processing blocks and how the data goes through those blocks. 
Interaction design: A description, following the "science of interaction" paper to be discussed in the lectures, of the high level and low level interactions you aim to have in the system and how they help in gaining insight (as defined in the lectures) in the problem or AI system you will consider. Provide a figure with the main components in your system and how they are connected through the interaction.  A simplified version of the main figure in the foundation model multimedia analytics paper (or the version discussed in the lectures for Multimedia Analytics for AI) could be a good starting point for this. 
Evaluation design:  How will you evaluate your solution following the different techniques in the evaluation lecture? Do you plan to recruit a few users or will you follow an analytic quality based approach using simulated actors? 
Implementation design: make a motivated choice for an architecture based on Dash / Plotly. To assure feasibiliy, it is recommended that you take (components of) the demo system provided as your starting point.
Planning: How will you realize your proposed solution in time for the demo and the deadline for the report? How do you assure a balance between delivering a solution that is innovative and at the same feasible to implement during the course of the project?
The evaluation of the  scientific report will be along  the following dimensions:

Motivation and Relevance:  Are all major decisions made in the report clearly motivated? Is the idea well embedded in the most relevant related work? Is it clear which target users would employ the system and why it would be relevant for them or if not in the application area what kind of systems would benefit from your new solution?

Complexity: How complex is the problem being addressed? And has the proposed solution the complexity that is needed to address the challenges? 

Implementation: Is there a clear design sketch, interaction scheme, and planning on how to create the system?

Scientific excellence: How innovative is the work with respect to what has already been published on the topic? 