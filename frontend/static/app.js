const searchBox = document.getElementById("search-box");
const searchResults = document.getElementById("search-results");
const recommendationsSection = document.getElementById("recommendations-section");
const selectedGameTitle = document.getElementById("selected-game-title");
const recommendationsGrid = document.getElementById("recommendations-grid");

let debounceTimer = null;

searchBox.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  const query = searchBox.value.trim();
  if (!query) {
    searchResults.innerHTML = "";
    return;
  }
  debounceTimer = setTimeout(() => runSearch(query), 200);
});

async function runSearch(query) {
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  const data = await res.json();
  renderSearchResults(data.games);
}

function renderSearchResults(games) {
  searchResults.innerHTML = "";
  for (const game of games) {
    const li = document.createElement("li");
    li.textContent = game.name;
    li.addEventListener("click", () => selectGame(game));
    searchResults.appendChild(li);
  }
}

async function selectGame(game) {
  searchResults.innerHTML = "";
  searchBox.value = game.name;

  const res = await fetch(`/api/games/${game.game_id}/recommendations`);
  const recommendations = await res.json();

  selectedGameTitle.textContent = `Because you like ${game.name}...`;
  recommendationsGrid.innerHTML = "";
  for (const rec of recommendations) {
    recommendationsGrid.appendChild(renderCard(rec));
  }
  recommendationsSection.hidden = false;
}

function renderCard(rec) {
  const card = document.createElement("div");
  card.className = "card";

  const title = document.createElement("h3");
  title.textContent = rec.game.name;

  const score = document.createElement("div");
  score.className = "score";
  score.textContent = `match score: ${(rec.score * 100).toFixed(0)}%`;

  const reasons = document.createElement("div");
  reasons.className = "reasons";
  reasons.textContent = rec.reasons.join(" · ");

  const genres = document.createElement("div");
  genres.className = "genres";
  if (rec.shared_genres.length) {
    genres.textContent = `Shared genres: ${rec.shared_genres.join(", ")}`;
  }

  card.append(title, score, reasons, genres);
  return card;
}
