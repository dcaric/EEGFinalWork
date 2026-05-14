1. Točan cilj projekta
Cilj projekta je predvidjeti količinu spavanja (prosječni sati sna) koristeći strojno učenje na temelju EEG podataka. Projekt nastoji utvrditi koji obrasci moždane aktivnosti (izraženi kroz FFT i sirove EEG signale) te metrike performansi (poput pažnje i angažmana) najbolje koreliraju s duljinom sna.
+4

2. Što točno profesor traži?
Profesor Sokač je definirao vrlo specifičan tehnički put:
Pilot faza: Započeti analizu s nasumičnih 10 ljudi kako bi se uspostavio cjevovod (pipeline) prije obrade cijelog skupa podataka.
Analiza značajki (FFT/Power vrijednosti):
Usporediti FFT vrijednosti među svim ispitanicima.
Analizirati metrike poput engagement, excitement i attention (iz data.csv) kroz različite zadatke (markere) za grupe ljudi koji spavaju 6, 7, 8 ili 9 sati.
Identificirati koji zadaci najbolje objašnjavaju razliku u satima spavanja.
Modeliranje u dvije faze:
Faza 1 (Klasični modeli): Izrada modela koji koristi FFT vrijednosti i demografske podatke (poput spola) za predviđanje sati sna. Mentor predlaže Random Forest i Boosting modele.
+2
Faza 2 (Sekvencijalni modeli): Korištenje sirovih (raw) EEG kanala kroz neuronske mreže (RNN, GRU, LSTM) na onim zadacima koji su se pokazali najvažnijima u prvoj fazi.
+1
Varijable iz DatasetSub_shorten (Tvoje pitanje o lijekovima i poremećajima):
Mentor izričito naglašava spol (Gender) kao ulazni parametar.
+1
Iako se "Povijest neuroloških poremećaja" i "Upotreba lijekova" ne nalaze na njegovom popisu "Variables of interest" , oni su prisutni u tvojoj tablici.
+1
Preporuka: S obzirom na to da takva stanja i lijekovi značajno utječu na EEG signal, svakako ih uključi u prve modele (Random Forest) kako bi vidjela njihovu važnost (feature importance). Ako imaju velik utjecaj, bit će važan dio tvoje rasprave u radu.

3. Plan rada (High-level plan)
I. Faza: Priprema i čišćenje podataka (Veljača)
Mapiranje podataka: Poveži ID osobe iz DatasetSub_shorten s njihovim pripadajućim data.csv i marker.csvdatotekama.
Sinkronizacija markera: Koristi timestamp iz obje datoteke kako bi označila segmente EEG podataka prema fazama (npr. eyesopen, eyesclose, Stimuli_phase).
Čišćenje: Provjeri kvalitetu signala pomoću stupaca EEG.Interpolated i CQ (Channel Quality) koje je mentor spomenuo.
II. Faza: Eksploratorna analiza (EDA) na Pilot grupi
Grupiranje po snu: Podijeli 10 pilota u grupe prema satima spavanja (npr. <6h, 7-8h, >8h).
Vizualizacija FFT-a: Izradi grafove za Power vrijednosti (Theta, Alpha, Beta) i usporedi ih kroz različite zadatke.
Dokumentacija: Odmah spremaj svaki graf i kratki opis; profesor je naglasio da će ti to drastično olakšati pisanje rada u lipnju.
III. Faza: Modeliranje - Klasični ML (Ožujak/Travanj)
Input: FFT vrijednosti + Spol + (opcionalno) Lijekovi/Poremećaji.
Modeli: Implementiraj Random Forest i Gradient Boosting koristeći Scikit-learn.
+1
Cilj: Utvrditi točnost predviđanja sati sna na temelju agregiranih podataka.
IV. Faza: Modeliranje - Deep Learning (Travanj/Svibanj)
Input: Sirovi EEG kanali kao vremenske serije.
Modeli: Implementiraj RNN, GRU ili LSTM mreže u PyTorch frameworku.
Fokus: Koristi samo podatke iz zadataka koji su se pokazali najinformativnijima u prethodnoj fazi.
V. Faza: Finalizacija i obrana (Lipanj/Srpanj)
Usporedba: Analiziraj jesu li kompleksni modeli na sirovim podacima dali bolje rezultate od klasičnih modela na FFT podacima.
Pisanje: Složi spremljene grafove i rezultate u predložak završenog rada.
Rokovi: Slanje zadnje verzije mentoru u lipnju kako bi obrana bila u srpnju

