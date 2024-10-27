// 1. Handle the Enter key press for the search
function handleKeyPress(event) {
    if (event.key === "Enter") {
        showResults();
    }
}

// 2. Show results when the search button is clicked
function showResults() {
    const searchInput = document.getElementById('search-input').value;
    sendSearchRequest(searchInput);
    smallBox.innerHTML = `
    <div class="circle-number">${index + 1}</div>
    <h2>${toolkit.name}</h2>
    <p>Domain: ${toolkit.domain}</p>
    <p>Framework: ${toolkit.framework}</p>
    <p>Programming Language: ${toolkit.programming_language}</p>
    <p>Role: ${toolkit.role}</p>
`;

}

// 3. Send request to the backend and get those juicy toolkits
function sendSearchRequest(searchInput) {
    fetch('/submit_search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `searchInput=${encodeURIComponent(searchInput)}` // Send the input value
    })
    .then(response => response.json())
    .then(recommendations => {
        const resultsScreen = document.getElementById('results-screen');
        const mainScreen = document.getElementById('main-screen');
        const toolkitList = document.querySelector('.toolkit-list');

        // Hide main screen and show results screen
        mainScreen.style.display = 'none';
        resultsScreen.style.display = 'flex';

        toolkitList.innerHTML = ''; // Clear previous results

        // Limit the recommendations to the first 6
        const limitedRecommendations = recommendations.slice(0, 6);
        
        // Populate the list with limited results
        limitedRecommendations.forEach((toolkit, index) => {
            const smallBox = document.createElement('div');
            smallBox.classList.add('small-box');
            smallBox.style.position = 'relative';
            smallBox.innerHTML = `
                <div class="circle-number">${index + 1}</div>
                <h2>${toolkit.name}</h2>
                <p>Domain: ${toolkit.domain}</p>
                <p>Framework: ${toolkit.framework}</p>
                <p>Programming Language: ${toolkit.programming_language}</p>
                <p>Role: ${toolkit.role}</p>
            `;
            smallBox.addEventListener('click', function() {
                showToolkitDetails(toolkit);
            });
            toolkitList.appendChild(smallBox);
        });

        // Debugging output to verify the results are being populated
        console.log("Recommendations:", limitedRecommendations);
    })
    .catch(error => console.error('Error:', error));
}



// 4. Filter toolkits on the second screen
function filterToolkits() {
    const input = document.getElementById('toolkit-search-input').value.toLowerCase();
    
    if (input.trim() === "") {
        const toolkits = document.querySelectorAll('.small-box');
        toolkits.forEach(toolkit => toolkit.style.display = 'block');
    } else {
        sendSearchRequest(input); // Trigger new search if input is not empty
    }
}

// 5. Show toolkit details when a small box is clicked
function showToolkitDetails(toolkit) {
    document.getElementById('results-screen').style.display = 'none';
    document.getElementById('detail-screen').style.display = 'flex';

    // Update the detail box content with better formatting
    document.getElementById('detail-name').innerHTML = `<h1>${toolkit.name || 'Toolkit Name'}</h1>`;
    document.getElementById('detail-domain').innerHTML = `<strong>Domain:</strong> ${toolkit.domain || 'N/A'}`;
    document.getElementById('detail-framework').innerHTML = `<strong>Framework:</strong> ${toolkit.framework || 'N/A'}`;
    document.getElementById('detail-role').innerHTML = `<strong>Role:</strong> ${toolkit.role || 'N/A'}`;
    document.getElementById('detail-description').innerHTML = `<strong>Description:</strong> ${toolkit.description || 'N/A'}`;
    
    // Add author name with click event
    document.getElementById('detail-author').innerHTML = `<strong>Author:</strong> <span class="clickable-author" onclick="showTicketRequestNotification('${toolkit.author || 'N/A'}')">${toolkit.author || 'N/A'}</span>`;
}

// 6. Back to results from details view
function backToResults() {
    document.getElementById('results-screen').style.display = 'flex';
    document.getElementById('detail-screen').style.display = 'none';
}

function showTicketRequestNotification(authorName) {
    const confirmation = confirm(`Send request ticket to ${authorName}?`);
    if (confirmation) {
        // Handle ticket request submission here
        console.log('Ticket request sent to:', authorName);
    } else {
        console.log('Ticket request canceled.');
    }
}



