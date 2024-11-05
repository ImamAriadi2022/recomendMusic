import React, { useState } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css"; // Pastikan Anda mengimpor Bootstrap
import "./App.css"; // Impor file CSS khusus untuk kustomisasi tambahan

function App() {
    const [mood, setMood] = useState([0, 0, 0]); // Inisialisasi mood dengan nilai default
    const [songs, setSongs] = useState([]); // Menyimpan lagu rekomendasi
    const [showSongs, setShowSongs] = useState(false); // Menyimpan status apakah daftar lagu ditampilkan atau tidak

    const handleSliderChange = (index, value) => {
        const newMood = [...mood];
        newMood[index] = parseFloat(value); // Memastikan value menjadi float
        setMood(newMood); // Update mood dengan nilai baru
    };

    const getRecommendations = async () => {
        if (mood.some(isNaN)) { // Memeriksa apakah ada nilai yang tidak valid
            alert("Please ensure all mood values are valid numbers.");
            return;
        }

        try {
            const response = await axios.post("http://localhost:5000/recommend", {
                mood: mood, // Mengirimkan array mood
            });
            setSongs(response.data); // Mengupdate state dengan lagu rekomendasi
            setShowSongs(true); // Tampilkan daftar lagu setelah mendapatkan rekomendasi
        } catch (error) {
            console.error("Error fetching recommendations:", error);
        }
    };

    const clearSongs = () => {
        setSongs([]); // Menghapus daftar lagu
        setShowSongs(false); // Menyembunyikan daftar lagu
    };

    return (
        <div className="d-flex justify-content-center align-items-center min-vh-100 bg-dark text-light p-4"> {/* Menambahkan kelas min-vh-100 */}
            <div className="container text-center">
                <h1 className="mb-4">Rekomendasi Musik</h1>

                {/* Slider untuk input mood dengan kategori relevan */}
                <div className="mb-4">
                    <div className="mb-3">
                        <label className="form-label">Tingkat Stres:</label>
                        <input
                            type="range"
                            className="form-range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={mood[0]} // Menampilkan nilai default
                            onChange={(e) => handleSliderChange(0, e.target.value)}
                        />
                        <span>{mood[0].toFixed(1)}</span> {/* Menampilkan nilai */}
                    </div>

                    <div className="mb-3">
                        <label className="form-label">Tingkat Kebahagiaan:</label>
                        <input
                            type="range"
                            className="form-range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={mood[1]} // Menampilkan nilai default
                            onChange={(e) => handleSliderChange(1, e.target.value)}
                        />
                        <span>{mood[1].toFixed(1)}</span> {/* Menampilkan nilai */}
                    </div>

                    <div className="mb-3">
                        <label className="form-label">Tingkat Energi:</label>
                        <input
                            type="range"
                            className="form-range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={mood[2]} // Menampilkan nilai default
                            onChange={(e) => handleSliderChange(2, e.target.value)}
                        />
                        <span>{mood[2].toFixed(1)}</span> {/* Menampilkan nilai */}
                    </div>
                </div>

                <button className="btn btn-outline-light" onClick={getRecommendations}>
                    Mencari Rekomendasi
                </button>
                
                <button className="btn btn-danger ms-2" onClick={clearSongs}>
                    Clear List
                </button>

                {/* Tampilkan daftar lagu jika ada */}
                {showSongs && (
                    <div className="mt-4">
                        <h2>List Musik:</h2>
                        <ul className="list-group list-group-flush">
                            {songs.map((song, index) => (
                                <li key={index} className="list-group-item bg-secondary text-light">
                                    {song.track_name} by {song.artist_name}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
