Evo all data. Svaka osoba ima data.csv i marker.csv iako svi markeri su isti. 
Ne trebas raditi sa cijelim datasetom, preporucam da krenes prvo sa random 10 ljudi pa onda onda expandas analizu kad sve radi ovako. 
 
-Prvo usporedi FFT (Pow vrijednosti) to mozes sa svima
-Onda npr kako kako izgleda engament,excitment, attention (vidjet ces te kolone u data.csv) razlicitih taskova ljudi koji spavaju 6,7,8,9 sati
-Da li FFT od nekih taskova bolje objasnjava razliku u sati spavanja
-Napravit model koji uzima FFT vrijednosti i proba predictat hours of sleep (mozes uzet gender kao input parameter)
-Spremi grafove i mali opis grafa negdje pa ce ti lako kasnije biti sve to bacit u zavrsni
-Probaj predictat hours of sleep na temelju raw EEG channels ali uzmi samo taskove koje si pronasla u 3. bulletpointu
 
Onda se cujemo za onaj drugi dio sa modeling raw EEGa
 
dr.sc. Mateo Sokač,
Assistant Professor
 
 
mateo.sokac@algebra.hr
 
Algebra
Gradišćanska 24, 10000 Zagreb
 
 
 
________________________________________
From: Paola Carić | Student <pcaric@algebra.hr>
Sent: Thursday, December 11, 2025 6:24 PM
To: Mateo Sokač <Mateo.Sokac@algebra.hr>
Subject: Re: Molba za mentorstvo
 
Oke, super hvala!
Ne, nisam dobila marker.csv.
Koliko vidim fale mi još ove 3 stvari da mogu započeti?
Ostatak Dataset-a (Raw CSV-ovi): Trenutno imam samo jednu datoteku(Zrinka).
Metadata tablica : Podatci o avg.hours of sleep za pojedinu osobu
Marker.csv
 
 
Sent from Outlook for Mac
 
From: Mateo Sokač <Mateo.Sokac@algebra.hr>
Date: Thursday, 11 December 2025 at 11:35
To: Paola Carić | Student <pcaric@algebra.hr>
Subject: Re: Molba za mentorstvo
Tako je, EEG su Raw, pow su FFT i nije ti loše uzet quality kolone (po kanalu i overal)
 
 
Nisam poslao marker.csv?
 
Može, možemo sve fokusirati u AVG. Hours of sleep. 
 
Ovo gdje sugeriras temu je opis za mene, možeš ostaviti prazno jer znamo koja je tema.
 
dr.sc. Mateo Sokač,
Assistant Professor 
 
 
 
mateo.sokac@algebra.hr
 
Algebra
Gradišćanska 24, 10000 Zagreb
 
 
________________________________________
From: Paola Carić | Student <pcaric@algebra.hr>
Sent: Monday, December 8, 2025 9:57:37 PM
To: Mateo Sokač <Mateo.Sokac@algebra.hr>
Subject: Re: Molba za mentorstvo
 
Je radi .zip….pogledala sam podatke i kako bi trebalo tu analizu napraviti. Kolko sam skužila trebaju mi samo  .EEG i .POW stupci? Vidim stupac 'MarkerValueInt' s vrijednostima poput 87147, 87148 itd. Ima li možda legendu (šifrarnik) što koji marker predstavlja, kako bih znala koje stanje klasificiram?
Od ovih variable of interest mi najbolje zvuči  Avg. hours of sleeping pa bih to koristila….
Također, na ovaj formular na infoeduci gdje biram mentora/temu….jeli mogu napisati nešto generalno za sad pa kasnije kad šaljemo komisiji u detalje?
 
 
Sent from Outlook for Mac
 
From: Mateo Sokač <Mateo.Sokac@algebra.hr>
Date: Monday, 8 December 2025 at 10:11
To: Paola Carić | Student <pcaric@algebra.hr>
Subject: Re: Molba za mentorstvo
Tako je pronaci koji model najbolje radi + koristeci EEG predicat jednu od onih variables of interest... npr Spol/Education level, itd. I onda vidjeti razlike koje su u EEGu.


Jel ovaj .zip radi?
 
dr.sc. Mateo Sokač,
Assistant Professor
 
 
mateo.sokac@algebra.hr
 
Algebra
Gradišćanska 24, 10000 Zagreb
 
 
 
________________________________________
From: Paola Carić | Student <pcaric@algebra.hr>
Sent: Friday, December 5, 2025 1:48 PM
To: Mateo Sokač <Mateo.Sokac@algebra.hr>
Subject: Re: Molba za mentorstvo
 
E super, hvala! Budem onda pogledala to sve detaljno kroz ovaj vikend pa se javim...
Razumijem da će ovo tražiti obradu EEG podataka i da ću za to trebati istražiti razne modele i naći onaj koji najbolje odgovara, ako sam dobro razumjela? Ako je to točno zanima me koji je krajni cilj projekta tj. što on mora da pokaže ili dokaže na kraju?
 
Btw nemogu otvorit zip., ima samo 22 byta kad downloadam.
 
________________________________________
From: Mateo Sokač <Mateo.Sokac@algebra.hr>
Sent: Friday, December 5, 2025 12:30 PM
To: Paola Carić | Student <pcaric@algebra.hr>
Subject: Re: Molba za mentorstvo
 
Nastavak na prosli email


Kako izgleda EEG data:

Variables of interest:
-Gender
-Education level
-Education field
-Languages speak
-Used to do sports (almost professional)
-Now doing sports a week
-Playing an instrument
-Avg. hours of sleeping
 
dr.sc. Mateo Sokač,
Assistant Professor
 
 
mateo.sokac@algebra.hr
 
Algebra
Gradišćanska 24, 10000 Zagreb
 
 
 
________________________________________
From: Mateo Sokač <Mateo.Sokac@algebra.hr>
Sent: Friday, December 5, 2025 12:26 PM
To: Paola Carić | Student <pcaric@algebra.hr>
Subject: Re: Molba za mentorstvo
 
Evo summary:

Data koji sam ti proposeao je EEG sto znaci da je time series  ili sequential (tj rows imaju time dependency).
Neka ideja zavrsnog je da istrazis prvo klasince modele koji ne assumaju time komponentu npr (Random Forest, Boosting, Logistic reg, itd)
Taj dio nebi previse radili jer znamo da nisu napravljeni za takav data pa idemo u python (pytorch) framework koji nam daje mogucnost implementiranja
Neuronskih mreza. Specificno fokusirali bih se na RNN, GRU i LSTM. Ako ima vremena Convolucija, specificno Conv1D. 

Ono sto bih trebala sto prije krenula raditi jest:
-Kako rade random forest, boosting i logistic regression modeli
-Sto je pytroch, sto nam omogucuje. Linearna al gebra u pozadini MLa
-Kako rade RNN, GRU i LSTM ( kad skuzis jednog, znat ces i druga dva :) )
-Kako implementirati i istrenirati model u pytrochu

Ono sto me ne brine previse:
-Scikit learn tj library za random forest, boosting i logistic regression jer su to vecinom 2,3 linije koda


Ono sto trebas jos saznati:
-Lektura 
-Diploma = Dokaz za Engleski jezik?
-Goldsmiths = University?
-Dal imas vremena za courseru. npr uzet neki Specialization od 5 courseva (Deeplearning.ai npr)


Pitao sam malo za datume za obranu. Ono sto smo gledali su bili datumi ako kreces sa zavrsnim u ovom semestru. Ti kreces zapravo u 2mj sto znaci da ti je prba moguca obrana 
vjv negdje u 7mj, sto znaci da meni saljes zadnju verziju u 6mj. 


Materijali:
Jako dobra knjiga za ML. Generalni koncepti, zasto kako itd
https://docdrop.org/download_annotation_doc/AAAMLP-569to.pdf

Pytorch ima jako dobru dokumentaciju tako da nebi sad trazio neku knjigu
https://pytorch.org/

Kako radi decision tree, random forest, boosting -> Stat Quest kanal na youtubeu
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ

RNN,GRU i LSTM -> Isto stat quest kanal na yt i takoder 3Blue1Brown kanal. 
Ovaj set videa je jako dobar za neuronske mreze opcenito:
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

DataScienceCamp je isto dobra stranica za vidjeti razne tutoriale itd.

I finalno, mozemo se vidjeti jednom u 2tj ili kako ce nam vise odgovarati.


Order of events za zavrsni:
1. Infoeduka mentorstvo
2. Istrazivanje i definiranje teme
3. Onaj template za zavrsni
4. WE GO HARD OR DIE TRYING 🙂
5. Saljemo zavrsni komisiji (approx sredina 6mj)
6. Komisija odgovara -> Prepravci + lektura
7. Saljemo popravke i dokaz o lekturi
Ovdje negdje moras prijaviti ispiti (tjedan dva dana prije obrane, kad god ti otvore u IE)
8. Obrana (mozemo probat pocetak 7mj pozuriti) 