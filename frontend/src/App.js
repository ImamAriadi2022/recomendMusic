// Import dependencies
import React, { useState } from "react";
import songsData from "./data/SpotifySample100.json"; // Import static song data
import "bootstrap/dist/css/bootstrap.min.css"; // Import Bootstrap for styling
import "./App.css"; // Import custom CSS

function App() {
    const [mood, setMood] = useState([0, 0, 0]); // Initialize mood state
    const [recommendedSongs, setRecommendedSongs] = useState([]); // State to store recommended songs
    const [showRecommendations, setShowRecommendations] = useState(false); // State to toggle song list display

    // Handle slider change for each mood parameter
    const handleSliderChange = (index, value) => {
        const updatedMood = [...mood];
        updatedMood[index] = parseFloat(value); // Ensure the value is a float
        setMood(updatedMood); // Update mood with new value
    };

    // Generate song recommendations based on mood
    const getRecommendations = () => {
        if (mood.some(isNaN)) { // Check if any mood value is invalid
            alert("Please ensure all mood values are valid numbers.");
            return;
        }

        // Filter songs based on mood criteria
        const filteredSongs = songsData.filter(song => {
            const moodMatch = 
                Math.abs(song.valence - mood[1]) < 0.2 && // Match on happiness (valence)
                Math.abs(song.energy - mood[2]) < 0.2 &&  // Match on energy
                Math.abs(song.acousticness - mood[0]) < 0.2; // Match on stress (acousticness)
            return moodMatch;
        });

        setRecommendedSongs(filteredSongs.slice(0, 10)); // Limit to top 10 recommendations
        setShowRecommendations(true); // Show song list after filtering
    };

    // Clear song recommendations
    const clearRecommendations = () => {
        setRecommendedSongs([]);
        setShowRecommendations(false);
    };

    return (
        <div className="d-flex justify-content-center align-items-center min-vh-100 bg-dark text-light p-4">
            <div className="container text-center">
                <h1 className="mb-4">Rekomendasi Musik Berdasarkan Mood</h1>

                {/* Mood input sliders */}
                <div className="mb-4">
                    <div className="mb-3">
                        <label className="form-label">Tingkat Stres:</label>
                        <input
                            type="range"
                            className="form-range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={mood[0]}
                            onChange={(e) => handleSliderChange(0, e.target.value)}
                        />
                        <span>{mood[0].toFixed(1)}</span>
                    </div>

                    <div className="mb-3">
                        <label className="form-label">Tingkat Kebahagiaan:</label>
                        <input
                            type="range"
                            className="form-range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={mood[1]}
                            onChange={(e) => handleSliderChange(1, e.target.value)}
                        />
                        <span>{mood[1].toFixed(1)}</span>
                    </div>

                    <div className="mb-3">
                        <label className="form-label">Tingkat Energi:</label>
                        <input
                            type="range"
                            className="form-range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={mood[2]}
                            onChange={(e) => handleSliderChange(2, e.target.value)}
                        />
                        <span>{mood[2].toFixed(1)}</span>
                    </div>
                </div>

                {/* Buttons for generating recommendations and clearing list */}
                <button className="btn btn-outline-light" onClick={getRecommendations}>
                    Cari Rekomendasi
                </button>
                
                <button className="btn btn-danger ms-2" onClick={clearRecommendations}>
                    Hapus List
                </button>

                {/* Display song recommendations if available */}
                {showRecommendations && recommendedSongs.length > 0 ? (
                    <div className="mt-4">
                        <h2>List Musik Rekomendasi:</h2>
                        <ul className="list-group list-group-flush">
                            {recommendedSongs.map((song, index) => (
                                <li key={index} className="list-group-item bg-secondary text-light">
                                    {song.track_name} by {song.artist_name}
                                </li>
                            ))}
                        </ul>
                    </div>
                ) : showRecommendations && (
                    <p className="mt-4 text-warning">Tidak ada rekomendasi yang cocok dengan mood saat ini.</p>
                )}
            </div>
        </div>
    );
}

export default App;
