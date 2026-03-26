function renderMovies(movies) {
    const resultsList = document.getElementById("resultsList");
    resultsList.innerHTML = "";

    const fragment = document.createDocumentFragment();

    movies.forEach(movie => {
        const div = document.createElement("div");
        div.className = "movie-card";
        div.innerHTML = `
            <img src="${movie.poster}" alt="${movie.title}" class="poster">
            <div class="movie-info">
                <h3>${movie.title} (${movie.year})</h3>
            </div>
        `;
        div.addEventListener("click", () => showMovieModal(movie));
        fragment.appendChild(div);
    });

    resultsList.appendChild(fragment);
}


function showMovieModal(movie) {
    const modal = document.getElementById("movieModal");
    const modalBody = document.getElementById("modalBody");

    modalBody.innerHTML = `
        <h2>${movie.title} (${movie.year})</h2>
        <img src="${movie.poster}" alt="${movie.title}" style="width:200px; float:left; margin-right:20px;">
        <p><strong>Genres:</strong> ${movie.genre}</p>
        <p><strong>Recommendation Score:</strong> ${movie.score}</p>
        <p><strong>Rating:</strong> ${(movie.rating_bayesian * 10).toFixed(1)}/10 (${movie.imdb_votes} votes)</p>
        <p><strong>Language:</strong> ${movie.primary_language}</p>
        <p><strong>Runtime:</strong> ${movie.runtime_mins}</p>
        <p><strong>Actors:</strong> ${movie.actors}</p>
        <p><strong>Plot:</strong> ${movie.plot}</p>
        <div style="clear: both;"></div>
    `;

    modal.style.display = "flex";

    document.getElementById("closeModal").onclick = () => {
        modal.style.display = "none";
    }

    modal.onclick = (e) => {
        if (e.target === modal) modal.style.display = "none";
    };

}


let globalLimit = 20
async function applyFilters(limit = globalLimit) {
    const params = new URLSearchParams({ limit });

    ['genreFilter', 'languageFilter', 'countryFilter', 'sort_by'].forEach(id => {
        const el = document.getElementById(id);
        const val = el?.value;
        if (val) params.append(id.replace("Filter",""), val);
    });

    const res = await fetch(`http://127.0.0.1:8000/recommendations?${params}`);
    const data = await res.json();

    renderMovies(data.results);
    populateFilters(
        data.all_genres, 
        data.all_languages, 
        data.all_countries, 
        data.sort_options,
        data.default_sort_option
    );

    const loadMoreBtn = document.getElementById("loadMore");
    loadMoreBtn.style.display = data.more_results_available ? "block" : "none";
}


function populateFilter(selectEl, items, allText, selectedValue) {
    selectEl.innerHTML = `<option value="">${allText}</option>`;
    items.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        selectEl.appendChild(opt);
    });
    if (selectedValue) selectEl.value = selectedValue;
}

function populateFilters(all_genres, all_languages, all_countries, sort_options, default_sort_option) {
    const genreSet = new Set(all_genres.filter(Boolean));
    const languageSet = new Set(all_languages.filter(Boolean));
    const countrySet = new Set(all_countries.filter(Boolean));
    const sortBySet = new Set(sort_options.filter(s => s && s !== default_sort_option));

    populateFilter(document.getElementById("genreFilter"), genreSet, "All Genres", document.getElementById("genreFilter").value);
    populateFilter(document.getElementById("languageFilter"), languageSet, "All Languages", document.getElementById("languageFilter").value);
    populateFilter(document.getElementById("countryFilter"), countrySet, "All Countries", document.getElementById("countryFilter").value);
    populateFilter(document.getElementById("sort_by"), sortBySet, default_sort_option, document.getElementById("sort_by").value);
}


async function resetFilters() {
    document.getElementById("languageFilter").value="";
    document.getElementById("genreFilter").value="";
    document.getElementById("countryFilter").value="";

    const sortByFilter = document.getElementById("sort_by");
    sortByFilter.value = sortByFilter.options[0].value;

    globalLimit = 20; 
    applyFilters();
}


async function loadMore() {
    globalLimit += 20;
    await applyFilters(globalLimit);
}


window.addEventListener("DOMContentLoaded", () => {
    ['genreFilter', 'languageFilter', 'countryFilter', 'sort_by'].forEach(el => {
        document.getElementById(el).addEventListener("change", () => applyFilters());
    })

    document.getElementById("reset").addEventListener("click", () => resetFilters());
    document.getElementById("loadMore").addEventListener("click", () => loadMore());

    applyFilters();
});
