# TODO

store all data in data folder


fix the gitignore

integrate text descriptions dataset








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


 # CBS WFS fields

The CBS socio-economic variables are loaded directly from the PDOK
`wijkenbuurten:buurten` WFS (no extra download — `buurten()` already fetches them).
To change which variables the system uses, edit the `CBS` dict in `precompute.py`:
each entry is `"column_label": "wfsFieldName"`. Pick any field below and re-run
`precompute.py`. `app.py` shows whatever labels you defined in the spider/table.

Caveats when picking fields:
- `afstandTot…` / `…GemiddeldeAfstandInKm` are **proximity** (km), not amount — a
  *smaller* value means *closer*.
- There is **no `% green`** field and **no `avg_age`** field — only distances to
  green/nature and the age-bucket percentages (`percentagePersonen…Jaar`).
- Some buurten (water-only / very small) have missing values; `build()` masks
  negatives and imputes the median so standardisation stays well-defined.

Full list of available fields on each buurt feature (PDOK CBS wijkenbuurten 2023):

**Demographics & population**
- `aantalInwoners` — number of inhabitants
- `mannen` — number of men
- `vrouwen` — number of women
- `bevolkingsdichtheidInwonersPerKm2` — population density (inhabitants/km²)
- `omgevingsadressendichtheid` — surrounding address density
- `stedelijkheidAdressenPerKm2` — urbanity (addresses/km²)

**Age distribution**
- `percentagePersonen0Tot15Jaar` — % persons 0–15 years
- `percentagePersonen15Tot25Jaar` — % persons 15–25 years
- `percentagePersonen25Tot45Jaar` — % persons 25–45 years
- `percentagePersonen45Tot65Jaar` — % persons 45–65 years
- `percentagePersonen65JaarEnOuder` — % persons 65+ years

**Marital status**
- `percentageOngehuwd` — % unmarried
- `percentageGehuwd` — % married
- `percentageGescheid` — % divorced
- `percentageVerweduwd` — % widowed

**Birth & death**
- `geboorteTotaal` — total births
- `geboortesPer1000Inwoners` — births per 1000 inhabitants
- `sterfteTotaal` — total deaths
- `sterfteRelatief` — relative death rate

**Households**
- `aantalHuishoudens` — number of households
- `percentageEenpersoonshuishoudens` — % single-person households
- `percentageHuishoudensZonderKinderen` — % households without children
- `percentageHuishoudensMetKinderen` — % households with children
- `gemiddeldeHuishoudsgrootte` — average household size

**Ethnicity & origin**
- `percentageMetHerkomstlandNederland` — % with Dutch origin
- `percentageMetHerkomstlandUitEuropaExclNl` — % origin elsewhere in Europe (excl. NL)
- `percentageMetHerkomstlandBuitenEuropa` — % origin outside Europe
- `percentageGebInNlMetHerkomstlandNederland` — % born in NL, Dutch origin
- `percGebInNlMetHerkomstlandInEuropaExNl` — % born in NL, European origin (excl. NL)
- `percGebInNlMetHerkomstlandBuitenEuropa` — % born in NL, non-European origin
- `percGebBuitenNlMetHerkomstlndInEuropaExNl` — % born outside NL, European origin (excl. NL)
- `percGebBuitenNlMetHerkomstlndBuitenEuropa` — % born outside NL, non-European origin

**Business & employment**
- `aantalBedrijfsvestigingen` — total business establishments
- `aantalBedrijvenLandbouwBosbouwVisserij` — businesses in agriculture/forestry/fishing
- `aantalBedrijvenNijverheidEnergie` — businesses in industry/energy
- `aantalBedrijvenHandelEnHoreca` — businesses in trade/hospitality
- `aantalBedrijvenVervoerInformatieCommunicatie` — businesses in transport/IT/communication
- `aantalBedrijvenFinancieelOnroerendGoed` — businesses in finance/real estate
- `aantalBedrijvenZakelijkeDienstverlening` — business-service companies
- `aantalBedrijvenOverheidOnderwijsEnZorg` — businesses in government/education/health
- `aantalBedrijvenCultuurRecreatieOverige` — businesses in culture/recreation/other
- `nettoArbeidsparticipatie` — net labour-force participation
- `percentageWerknemers` — % employees
- `percentageZelfstandigen` — % self-employed
- `aantalPersWerkzameBeroepsbevolking` — employed working population
- `percentageWerknemersMetVasteArbeidsrelatie` — % with permanent employment
- `percentageWerknemersMetFlexibeleArbeidsrelatie` — % with flexible employment

**Housing & real estate**
- `gemiddeldeWoningwaarde` — average home value (WOZ)
- `woningvoorraad` — housing stock (total dwellings)
- `percentageEengezinswoning` — % single-family homes
- `percentageMeergezinswoning` — % multi-family homes
- `percentageBewoond` — % occupied dwellings
- `percentageOnbewoond` — % unoccupied dwellings
- `percentageLeegstandWoningen` — % vacant dwellings
- `percentageKoopwoningen` — % owner-occupied dwellings
- `percentageHuurwoningen` — % rental dwellings
- `percHuurwoningenInBezitWoningcorporaties` — % rentals owned by housing corporations
- `percHuurwoningenInBezitOverigeVerhuurders` — % rentals owned by other landlords
- `percentageWoningenMetEigendomOnbekend` — % dwellings with unknown ownership
- `percentageBouwjaarklasseTot2000` — % built before 2000
- `percentageBouwjaarklasseVanaf2000` — % built from 2000 onwards

**Energy consumption**
- `gemiddeldGasverbruikTotaal` — average gas consumption (total)
- `gemiddeldGasverbruikAppartement` / `…Tussenwoning` / `…Hoekwoning` / `…2Onder1KapWoning` / `…VrijstaandeWoning` — gas by dwelling type
- `gemiddeldGasverbruikHuurwoning` / `gemiddeldGasverbruikkoopwoning` — gas by tenure
- `gemiddeldAardgasverbruik` — average natural-gas consumption
- `gemiddeldElektriciteitsverbruikTotaal` — average electricity consumption (total)
- `gemiddeldElektriciteitsverbruikAppartement` / `…Tussenwoning` / `…Hoekwoning` / `gemElektriciteitsverbruik2Onder1KapWoning` / `gemElektriciteitsverbruikVrijstaandeWoning` — electricity by dwelling type
- `gemiddeldElektriciteitsverbruikHuurwoning` / `gemiddeldElektriciteitsverbruikkoopwoning` — electricity by tenure
- `gemiddeldeElektriciteitslevering` — average electricity delivery
- `percentageWoningenMetStadsverwarming` — % homes with district heating

**Income & social benefits**
- `aantalInkomensontvangers` — number of income earners
- `gemiddeldInkomenPerInkomensontvanger` — average income per earner
- `gemiddeldInkomenPerInwoner` — average income per resident
- `gemiddeldGestandaardiseerdInkomenVanHuishoudens` — average standardised household income
- `mediaanVermogenVanParticuliereHuish` — median wealth of private households
- `percentagePersonenMetHoogInkomen` / `percentagePersonenMetLaagInkomen` — % persons high/low income
- `percentageHuishoudensMetHoogInkomen` / `percentageHuishoudensMetLaagInkomen` — % households high/low income
- `percentageHuishoudensMetLaagsteInkomen` — % households lowest income
- `percentageHuishoudensMetLageKoopkracht` — % households low purchasing power
- `percentageHuishoudensOnderOfRondSociaalMinimum` — % households at/below social minimum
- `huishoudensTot110PercentVanSociaalMinimum` / `huishoudensTot120PercentVanSociaalMinimum` — households up to 110%/120% of social minimum
- `aantalPersonenMetEenAoUitkeringTotaal` — persons on disability benefit
- `aantalPersonenMetEenWwUitkeringTotaal` — persons on unemployment benefit
- `aantalPersonenMetEenAlgBijstandsuitkeringTot` — persons on general assistance
- `aantalPersonenMetEenAowUitkeringTotaal` — persons on old-age pension

**Education**
- `opleidingsniveauLaag` / `opleidingsniveauMiddelbaar` / `opleidingsniveauHoog` — low/medium/high education-level counts
- `aantalLeerlingenPrimairOnderwijs` / `aantalLeerlingenVoortgezetOnderwijs` — primary/secondary pupils
- `aantalStudentenMbo` / `aantalStudentenHbo` / `aantalStudentenWo` — MBO/HBO/WO students

**Healthcare & social services**
- `huisartsenpraktijkGemiddeldeAfstandInKm` (+ `…GemiddeldAantalBinnen1Km/3Km/5Km`) — GP practice distance / counts
- `huisartsenpostGemiddeldeAfstandInKm` — distance to GP emergency post
- `apotheekGemiddeldeAfstandInKm` — distance to pharmacy
- `ziekenhuisInclBuitenpolikliniekGemAfstInKm` (+ `…GemAantalBinnen5Km/10Km/20Km`) — hospital (incl. outpatient) distance / counts
- `ziekenhuisExclBuitenpolikliniekGemAfstInKm` (+ within 5/10/20 km) — hospital (excl. outpatient) distance / counts
- `aantalJongerenMetJeugdzorgInNatura` / `percentageJongerenMetJeugdzorgInNatura` — youth in care (count / %)
- `aantalWmoClienten` / `aantalWmoClientenPer1000Inwoners` — WMO social-care clients (count / per 1000)

**Transport & vehicles**
- `personenautosTotaal` — total private cars
- `personenautosPerHuishouden` — cars per household
- `personenautosPerKm2` — cars per km²
- `motortweewielersTotaal` — total motorcycles
- `aantalPersonenautosMetBrandstofBenzine` — petrol cars
- `aantalPersonenautosMetOverigeBrandstof` — cars with other fuels
- `opritHoofdverkeerswegGemiddeldeAfstandInKm` — distance to motorway entrance
- `treinstationGemiddeldeAfstandInKm` — distance to train station
- `overstapstationGemiddeldeAfstandInKm` — distance to transit interchange
- `brandweerkazerneGemiddeldeAfstandInKm` — distance to fire station

**Land use & surface area**
- `oppervlakteTotaalInHa` — total area (ha)
- `oppervlakteLandInHa` — land area (ha)
- `oppervlakteWaterInHa` — water area (ha)

**Green space & nature** (all are *distances* in km, not percentages)
- `afstandTotOpenbaarGroenTotaal` — distance to public green space
- `afstandTotParkOfPlantsoen` — distance to park/garden
- `afstandTotBos` — distance to forest
- `afstandTotOpenNatuurTerreinTotaal` — distance to open nature area
- `afstandTotOpenDroogNatuurTerrein` — distance to open dry nature terrain
- `afstandTotOpenNatNatuurlijkTerrein` — distance to open wet nature terrain
- `afstandTotSemiopenbaarGroenTotaal` — distance to semi-public green space
- `afstandTotDagrcreatiefTerrein` — distance to day-recreation area
- `afstandTotVerblijfsrecreatiefTerrein` — distance to stay-recreation area
- `afstandTotRecreatiefBinnenwater` — distance to recreational inland water
- `afstandTotSportterrein` — distance to sports ground
- `afstandTotVolkstuin` — distance to allotment garden
- `afstandTotBegraafplaats` — distance to cemetery

**Retail & services** (distance + counts within N km)
- `groteSupermarktGemiddeldeAfstandInKm` (+ within 1/3/5 km) — large supermarket
- `winkelsOvDagelijkseLevensmGemAfstInKm` (+ within 1/3/5 km) — daily-goods shops
- `warenhuisGemiddeldeAfstandInKm` (+ within 5/10/20 km) — department store
- `cafeGemiddeldeAfstandInKm` (+ within 1/3/5 km) — café
- `cafetariaGemiddeldeAfstandInKm` (+ within 1/3/5 km) — cafeteria/snack bar
- `restaurantGemiddeldeAfstandInKm` (+ within 1/3/5 km) — restaurant
- `hotelGemiddeldeAfstandInKm` (+ within 5/10/20 km) — hotel

**Childcare & schools** (distance + counts within N km)
- `kinderdagverblijfGemiddeldeAfstandInKm` (+ within 1/3/5 km) — daycare
- `buitenschoolseOpvangGemAfstandInKm` (+ within 1/3/5 km) — after-school care
- `basisonderwijsGemiddeldeAfstandInKm` (+ within 1/3/5 km) — primary school
- `voortgezetOnderwijsGemAfstandInKm` (+ within 3/5/10 km) — secondary school
- `vmboGemiddeldeAfstandInKm` (+ within 3/5/10 km) — VMBO school
- `havoVwoGemiddeldeAfstandInKm` (+ within 3/5/10 km) — HAVO/VWO school

**Recreation & leisure** (distance, some with counts within N km)
- `zwembadGemiddeldeAfstandInKm` — swimming pool
- `kunstijsbaanGemiddeldeAfstandInKm` — ice rink
- `bibliotheekGemiddeldeAfstandInKm` — library
- `poppodiumGemiddeldeAfstandInKm` — pop venue
- `bioscoopGemiddeldeAfstandInKm` (+ within 5/10/20 km) — cinema
- `saunaGemiddeldeAfstandInKm` — sauna
- `zonnebankGemiddeldeAfstandInKm` — tanning salon
- `attractieparkGemiddeldeAfstandInKm` (+ within 10/20/50 km) — amusement park
- `theaterGemiddeldeAfstandInKm` (+ within 5/10/20 km) — theatre
- `gemiddeldeAfstandTotMuseum` (+ `gemiddeldAantalMuseaBinnen5Km/10Km/20Km`) — museum

**Administrative & metadata**
- `buurtcode` — neighbourhood code
- `buurtnaam` — neighbourhood name
- `wijkcode` — district code
- `gemeentecode` — municipality code
- `gemeentenaam` — municipality name
- `meestVoorkomendePostcode` — most common postcode
- `dekkingspercentage` — coverage percentage
- `water` — water indicator
- `jaar` — year (2023)

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



# text descriptions dataset instructions
Hi Jonas!

Fri, Jun 12, 2026 at 11:58

If you open the files, you will find multiple folders including a data summary markdown file that

explains the data.

The descriptions are in a .jsonl file of the following format (example). They describe tiles

(regions) and not singular panoramas.

{"patch_row": 27, "patch_col": 406, "canon_row": 5, "canon_col": 81, "quadrant": 11,

"describe_mode": "none", "mllm_output": "A wide expanse of fine, white sand stretches under a

blue sky with scattered clouds, its surface marked by gentle ripples and faint tracks. In the

distance, a low line of green vegetation separates the dunes from a flat horizon where a few

small figures stand. The ground is entirely covered in loose, pale sand with no signs of

pavement, structures, or cultivated earth. A single, narrow path cuts through the foreground,

leading toward the distant treeline. There are no buildings, roads, or human-made objects

visible anywhere in the view. The landscape remains unchanged across all captured

perspectives, showing only variations in camera angle and slight shifts in cloud cover. This is a

natural coastal dune system, devoid of development, preserving its raw, windswept character.”}

You can use canon_col and canon_row to know where they fit within the canonical_tiles.json

splitting of the country, and ‘quadrant’ to know exactly given a splitting. In the following link, you

can download also a tiles.paruqet file that I created which helps you map it in an easier way

(make sure you use your uva email to download it).

tiles.parquet

The description is the “mllm_output”, which is the text that is based on the gsv panoramas.

So now you have numerical values in the raster (gsv _analysis), and panorama images, and text

descriptions.

You have additionally climate data.

I hope this clarifies it.

Most details are in the markdown, and do not hesitate to ask me for further explanations.

Best regards,

Yahia Dalbah