#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include <mpi.h>
#include "contexte.hpp"
#include "individu.hpp"
#include "graphisme/src/SDL2/sdl2.hpp"
#include <omp.h>

void màjStatistique( épidémie::Grille& grille, std::vector<épidémie::Individu> const& individus )
{
    omp_set_num_threads(4);

    #pragma omp parallel for schedule(static) 
    for ( auto& statistique : grille.getStatistiques() )
    {
        statistique.nombre_contaminant_grippé_et_contaminé_par_agent = 0;
        statistique.nombre_contaminant_seulement_contaminé_par_agent = 0;
        statistique.nombre_contaminant_seulement_grippé              = 0;
    }
    auto [largeur,hauteur] = grille.dimension();
    auto& statistiques = grille.getStatistiques();

    #pragma omp parallel for schedule(static) 
    for ( auto const& personne : individus )
    {
        auto pos = personne.position();

        std::size_t index = pos.x + pos.y * largeur;
        if (personne.aGrippeContagieuse() )
        {
            if (personne.aAgentPathogèneContagieux())
            {
                statistiques[index].nombre_contaminant_grippé_et_contaminé_par_agent += 1;
            }
            else 
            {
                statistiques[index].nombre_contaminant_seulement_grippé += 1;
            }
        }
        else
        {
            if (personne.aAgentPathogèneContagieux())
            {
                statistiques[index].nombre_contaminant_seulement_contaminé_par_agent += 1;
            }
        }
    }
}

void afficheSimulation(sdl2::window& écran, épidémie::Grille const& grille, std::size_t jour)
{
    auto [largeur_écran,hauteur_écran] = écran.dimensions();
    hauteur_écran = hauteur_écran - 100;
    auto [largeur_grille,hauteur_grille] = grille.dimension();
    auto const& statistiques = grille.getStatistiques();
    sdl2::font fonte_texte("./graphisme/src/data/Lato-Thin.ttf", 18);
    écran.cls({0x00,0x00,0x00});
    // Affichage de la grille :
    std::uint16_t stepX = largeur_écran/largeur_grille;
    unsigned short stepY = (hauteur_écran-50)/hauteur_grille;
    double factor = 255./15.;
 
    for ( unsigned short i = 0; i < largeur_grille; ++i )
    {
        for (unsigned short j = 0; j < hauteur_grille; ++j )
        {
            auto const& stat = statistiques[i+j*largeur_grille];
            int valueGrippe = stat.nombre_contaminant_grippé_et_contaminé_par_agent+stat.nombre_contaminant_seulement_grippé;
            int valueAgent  = stat.nombre_contaminant_grippé_et_contaminé_par_agent+stat.nombre_contaminant_seulement_contaminé_par_agent;
            std::uint16_t origx = i*stepX;
            std::uint16_t origy = j*stepY;
            std::uint8_t red = valueGrippe > 0 ? 127+std::uint8_t(std::min(128., 0.5*factor*valueGrippe)) : 0;
            std::uint8_t green = std::uint8_t(std::min(255., factor*valueAgent));
            std::uint8_t blue= std::uint8_t(std::min(255., factor*valueAgent ));
            écran << sdl2::rectangle({origx,origy}, {stepX,stepY}, {red, green,blue}, true);
        }
    }

    écran << sdl2::texte("Carte population grippée", fonte_texte, écran, {0xFF,0xFF,0xFF,0xFF}).at(largeur_écran/2, hauteur_écran-20);
    écran << sdl2::texte(std::string("Jour : ") + std::to_string(jour), fonte_texte, écran, {0xFF,0xFF,0xFF,0xFF}).at(0,hauteur_écran-20);
    écran << sdl2::flush;
}

void simulation(bool affiche, int nbp, int rank)
{
    const int tag = 1;

    unsigned int graine_aléatoire = 1;
    std::uniform_real_distribution<double> porteur_pathogène(0.,1.);

    int num_thr = 4;

    épidémie::ContexteGlobal contexte;
    // contexte.déplacement_maximal = 1; <= Si on veut moins de brassage
    // contexte.taux_population = 400'000;
    //contexte.taux_population = 1'000;
    omp_set_num_threads(num_thr);
    #pragma omp parallel for  // partie 1 - enlever le for
    for (int k = 0; k < omp_get_num_threads(); k++) { // c'est stupide, je le sais, mais ça ne marche pas autrement :*(
        contexte.taux_population = 100000 * omp_get_num_threads();
    }
    contexte.interactions.β = 60.;
    std::vector<épidémie::Individu> population;
    population.reserve(contexte.taux_population);
    épidémie::Grille grille{contexte.taux_population};

    auto [largeur_grille,hauteur_grille] = grille.dimension();


    std::size_t jours_écoulés = 0;
    int         jour_apparition_grippe = 0;
    int         nombre_immunisés_grippe= (contexte.taux_population*23)/100;
    sdl2::event_queue queue;

    int quitting = 0;

    épidémie::Grippe grippe(0);
    MPI_Status Status;
    MPI_Status status2;
    MPI_Request req;
    if (rank == 1) {
       
        // L'agent pathogène n'évolue pas et reste donc constant...
        épidémie::AgentPathogène agent(graine_aléatoire++);
        // Initialisation de la population initiale :

        auto start_sim = std::chrono::system_clock::now();
        auto end_sim = std::chrono::system_clock::now();
        std::chrono::duration<double> delta_sim = end_sim - start_sim;

        auto start_glob = std::chrono::system_clock::now();

        //int compteur = 0;
        //omp_set_num_threads(num_thr);
        //#pragma omp parallel for schedule(static) 
        for (std::size_t i = 0; i < contexte.taux_population; ++i) // on peut paralléliser cette boucle, mais je n'arrive pas à obtenir des mêmes résultats ... 
        {
            //int tid = omp_get_thread_num();
            //if (compteur == 0) // pour le faire une seul fois
            //{
            //    graine_aléatoire += (num_thr-tid)*contexte.taux_population/num_thr; // ça dépend de l'ordre, donc il faut ordonnancer les graine comme ne séquentiel
            //}
            std::default_random_engine motor(100 * (i + 1));            
            //#pragma omp critical
            population.emplace_back(graine_aléatoire++, contexte.espérance_de_vie, contexte.déplacement_maximal);
            population.back().setPosition(largeur_grille, hauteur_grille);
            if (porteur_pathogène(motor) < 0.2)
            {
                population.back().estContaminé(agent);
            }
            //compteur += 1;
        }

        std::cout << "Début boucle épidémie" << std::endl << std::flush;
        quitting = false;
        while (!quitting)
        {
            start_sim = std::chrono::system_clock::now();
            auto events = queue.pull_events();

            if (jours_écoulés % 365 == 0)// Si le premier Octobre (début de l'année pour l'épidémie ;-) )
            {
                grippe = épidémie::Grippe(jours_écoulés / 365);
                jour_apparition_grippe = grippe.dateCalculImportationGrippe();
                grippe.calculNouveauTauxTransmission();
                // 23% des gens sont immunisés. On prend les 23% premiers
                omp_set_num_threads(num_thr);
                #pragma omp parallel for schedule(static) 
                for (int ipersonne = 0; ipersonne < nombre_immunisés_grippe; ++ipersonne)
                {
                    population[ipersonne].devientImmuniséGrippe();
                }
                omp_set_num_threads(num_thr);
                #pragma omp parallel for schedule(static) 
                for (int ipersonne = nombre_immunisés_grippe; ipersonne < int(contexte.taux_population); ++ipersonne)
                {
                    population[ipersonne].redevientSensibleGrippe();
                }
            }
            if (jours_écoulés % 365 == std::size_t(jour_apparition_grippe))
            {
                omp_set_num_threads(num_thr);
                #pragma omp parallel for schedule(static) 
                for (int ipersonne = nombre_immunisés_grippe; ipersonne < nombre_immunisés_grippe + 25; ++ipersonne)
                {
                    population[ipersonne].estContaminé(grippe);
                }
            }
            // Mise à jour des statistiques pour les cases de la grille :
            màjStatistique(grille, population);
            // On parcout la population pour voir qui est contaminé et qui ne l'est pas, d'abord pour la grippe puis pour l'agent pathogène
            std::size_t compteur_grippe = 0, compteur_agent = 0, mouru = 0;
            omp_set_num_threads(num_thr);
            #pragma omp parallel for schedule(static) 
            for (auto& personne : population)
            {
                if (personne.testContaminationGrippe(grille, contexte.interactions, grippe, agent))
                {
                    compteur_grippe++;
                    personne.estContaminé(grippe);
                }
                if (personne.testContaminationAgent(grille, agent))
                {
                    compteur_agent++;
                    personne.estContaminé(agent);
                }
                // On vérifie si il n'y a pas de personne qui veillissent de veillesse et on génère une nouvelle personne si c'est le cas.
                if (personne.doitMourir())
                {
                    mouru++;
                    unsigned nouvelle_graine = jours_écoulés + personne.position().x * personne.position().y;
                    personne = épidémie::Individu(nouvelle_graine, contexte.espérance_de_vie, contexte.déplacement_maximal);
                    personne.setPosition(largeur_grille, hauteur_grille);
                }
                personne.veillirDUnJour();
                personne.seDéplace(grille);
            }

            // Envoyer : grille, jours      

            int flag = 0;

            MPI_Iprobe(0, 100500, MPI_COMM_WORLD, &flag, &status2);

            auto stats_grille = grille.getStatistiques();

            

            MPI_Request req_grille_principale;

            if(flag){
                MPI_Irecv(&quitting, 1, MPI_FLOAT, 0, 100500, MPI_COMM_WORLD, &req);

                MPI_Isend(stats_grille.data(), 3*(stats_grille.size()), MPI_INT, 0, tag, MPI_COMM_WORLD, &req_grille_principale);
                MPI_Isend(&jours_écoulés, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &req);

                MPI_Wait(&req_grille_principale, &status2); // faut attendre la confirmation de la réception
            }

            MPI_Bcast(&quitting, 1, MPI_INT, 1, MPI_COMM_WORLD);

            //std::cout << quitting << std::endl << std::flush;
            jours_écoulés += 1;
            end_sim = std::chrono::system_clock::now();
            delta_sim += (end_sim - start_sim);
            
        }// Fin boucle temporelle; simulation

        std::cout << "Temps de calcul de la simulation par jour : " << (delta_sim).count() / jours_écoulés << "s" << std::endl; 

        auto end_glob = std::chrono::system_clock::now();
        std::chrono::duration<double> delta_glob = end_glob - start_glob; // if rank == 0
        std::cout << "Temps de calcul global par jour : " << (delta_glob).count() / jours_écoulés << "s" << std::endl;
    }

    //#############################################################################################################
    //##    Affichage des résultats pour le temps  actuel
    //#############################################################################################################

    if(rank == 0){
        std::ofstream output("Courbe.dat");
        output << "# jours_écoulés \t nombreTotalContaminésGrippe \t nombreTotalContaminésAgentPathogène()" << std::endl;

        constexpr const unsigned int largeur_écran = 1280, hauteur_écran = 1024;
        sdl2::window écran("Simulation épidémie de grippe", { largeur_écran,hauteur_écran });

        auto start_aff = std::chrono::system_clock::now();
        auto end_aff = std::chrono::system_clock::now();
        std::chrono::duration<double> delta_aff = end_aff - start_aff;

        MPI_Status status;
        MPI_Request req;

        std::size_t jours_écoulés_loc = 0;
        
        auto stats_grille_loc = grille.getStatistiques();
        std::array<int, 2> dims_loc = grille.dimension();

        épidémie::Grille grille_loc{ contexte.taux_population };
        quitting = false;
        while (!quitting)
        {    
            start_aff = std::chrono::system_clock::now();

            //int flag = 0; // c'est faux, supprimmer cela plus tard
            //while(!flag){
            //    MPI_Iprobe(1, tag, MPI_COMM_WORLD, &flag, &status);
            //}

            auto events = queue.pull_events();
            for (const auto& e : events)
            {
                if (e->kind_of_event() == sdl2::event::quit)
                    quitting = 1;
            }
            MPI_Isend(&quitting, 1, MPI_INT, 1, 100500, MPI_COMM_WORLD, &req);

            // Recevoir : grille, jours
            
            MPI_Recv(stats_grille_loc.data(), 3 * (stats_grille_loc.size()), MPI_INT, 1, tag, MPI_COMM_WORLD, &status);

            MPI_Recv(&jours_écoulés_loc, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &status);

            grille_loc.setStatistiques(stats_grille_loc);

            grille_loc.setDimx(dims_loc[0]);
            grille_loc.setDimy(dims_loc[1]);
            
            if (affiche) afficheSimulation(écran, grille_loc, jours_écoulés_loc);
            end_aff = std::chrono::system_clock::now();
            delta_aff += (end_aff - start_aff);

            output << jours_écoulés << "\t" << grille_loc.nombreTotalContaminésGrippe() << "\t"
                   << grille_loc.nombreTotalContaminésAgentPathogène() << std::endl;


        }// Fin boucle temporelle; affichage

        output.close(); // if rank == 1 par ex.
        std::cout << "Temps de calcul de l'affichage par jour : " << (delta_aff).count() / jours_écoulés_loc << "s" << std::endl; // if rank == 1
    }
}

int main(int argc, char* argv[])
{
    int nbp, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nbp);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // parse command-line
    bool affiche = true;
    for (int i=0; i<argc; i++) {
      std::cout << i << " " << argv[i] << "\n";
      if (std::string("-nw") == argv[i]) affiche = false;
    }
  
    sdl2::init(argc, argv);
    {
        simulation(affiche, nbp, rank);
    }
    sdl2::finalize();

    MPI_Finalize();

    return EXIT_SUCCESS;
}
