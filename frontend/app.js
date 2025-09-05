// Agricultural AI Platform - Main Application JavaScript
// Integrates hyperspectral imaging, environmental sensors, and AI models

class AgriculturePlatform {
    constructor() {
        this.apiKey = 'AIzaSyApIAX0hMK5LKxEu5-pR590MaTw1YJ5Nsk';
        this.map = null;
        this.charts = {};
        this.sensorData = {};
        this.spectralData = {};
        this.aiModels = {};
        this.alertSystem = new AlertSystem();
        
        this.init();
    }

    async init() {
        try {
            console.log('Initializing Agricultural AI Platform...');
            
            // Initialize map
            this.initializeMap();
            
            // Initialize charts
            this.initializeCharts();
            
            // Load AI models
            await this.loadAIModels();
            
            // Start data streaming
            this.startDataStreaming();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Load initial data
            await this.loadInitialData();
            
            console.log('Platform initialized successfully');
        } catch (error) {
            console.error('Platform initialization failed:', error);
            this.showError('Failed to initialize platform. Please refresh the page.');
        }
    }

    // Map Initialization with Field Overlays
    initializeMap() {
        // Initialize Leaflet map
        this.map = L.map('field-map').setView([40.7128, -74.0060], 15);

        // Add satellite imagery layer
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }).addTo(this.map);

        // Add field boundaries and health zones
        this.addFieldBoundaries();
        this.addHealthZones();
        this.addSensorPoints();
    }

    addFieldBoundaries() {
        // Simulate field boundaries
        const fieldBoundary = [
            [40.7150, -74.0080],
            [40.7150, -74.0040],
            [40.7110, -74.0040],
            [40.7110, -74.0080],
            [40.7150, -74.0080]
        ];

        L.polygon(fieldBoundary, {
            color: '#059669',
            weight: 2,
            fillOpacity: 0.1
        }).addTo(this.map).bindPopup('Field 1: North Section (25 acres)');
    }

    addHealthZones() {
        // Simulate health zones with different NDVI values
        const healthZones = [
            { coords: [[40.7145, -74.0075], [40.7145, -74.0055], [40.7130, -74.0055], [40.7130, -74.0075]], health: 'excellent', ndvi: 0.85 },
            { coords: [[40.7130, -74.0075], [40.7130, -74.0055], [40.7115, -74.0055], [40.7115, -74.0075]], health: 'good', ndvi: 0.72 },
            { coords: [[40.7145, -74.0055], [40.7145, -74.0045], [40.7115, -74.0045], [40.7115, -74.0055]], health: 'fair', ndvi: 0.58 },
        ];

        healthZones.forEach((zone, index) => {
            const color = this.getHealthColor(zone.health);
            L.polygon(zone.coords, {
                color: color,
                weight: 1,
                fillColor: color,
                fillOpacity: 0.6
            }).addTo(this.map).bindPopup(`Zone ${index + 1}: ${zone.health.toUpperCase()}<br>NDVI: ${zone.ndvi}`);
        });
    }

    addSensorPoints() {
        // Add environmental sensor locations
        const sensorLocations = [
            { lat: 40.7140, lng: -74.0070, type: 'soil', status: 'active' },
            { lat: 40.7125, lng: -74.0060, type: 'weather', status: 'active' },
            { lat: 40.7135, lng: -74.0050, type: 'leaf_wetness', status: 'active' }
        ];

        sensorLocations.forEach(sensor => {
            const icon = this.getSensorIcon(sensor.type);
            L.marker([sensor.lat, sensor.lng], { icon })
                .addTo(this.map)
                .bindPopup(`${sensor.type.replace('_', ' ').toUpperCase()} Sensor<br>Status: ${sensor.status}`);
        });
    }

    getHealthColor(health) {
        const colors = {
            excellent: '#059669',
            good: '#84cc16',
            fair: '#f59e0b',
            poor: '#ef4444'
        };
        return colors[health] || '#6b7280';
    }

    getSensorIcon(type) {
        const iconMap = {
            soil: '🌱',
            weather: '🌡️',
            leaf_wetness: '💧'
        };
        
        return L.divIcon({
            html: `<div style="background: white; border-radius: 50%; padding: 4px; font-size: 16px;">${iconMap[type]}</div>`,
            iconSize: [30, 30],
            className: 'sensor-icon'
        });
    }

    // Charts Initialization
    initializeCharts() {
        this.initializeHealthChart();
        this.initializeEnvironmentChart();
        this.initializeSpectralChart();
    }

    initializeHealthChart() {
        const ctx = document.getElementById('healthChart').getContext('2d');
        this.charts.health = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
                datasets: [{
                    label: 'NDVI',
                    data: [0.65, 0.68, 0.72, 0.75, 0.78, 0.76, 0.80],
                    borderColor: '#059669',
                    backgroundColor: 'rgba(5, 150, 105, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'EVI',
                    data: [0.55, 0.58, 0.62, 0.65, 0.68, 0.66, 0.70],
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.4,
                        max: 1.0
                    }
                }
            }
        });
    }

    initializeEnvironmentChart() {
        const ctx = document.getElementById('environmentChart').getContext('2d');
        this.charts.environment = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
                datasets: [{
                    label: 'Soil Moisture (%)',
                    data: [72, 71, 69, 65, 68, 70, 72],
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 50,
                        max: 100
                    }
                }
            }
        });
    }

    initializeSpectralChart() {
        const ctx = document.getElementById('spectralChart').getContext('2d');
        
        // Generate spectral signature data
        const wavelengths = [];
        const reflectance = [];
        
        for (let i = 400; i <= 1000; i += 10) {
            wavelengths.push(i);
            // Simulate typical vegetation spectral signature
            let value;
            if (i < 700) {
                value = 0.05 + Math.random() * 0.1; // Low reflectance in visible
            } else if (i < 750) {
                value = 0.1 + (i - 700) / 50 * 0.6; // Red edge
            } else {
                value = 0.7 + Math.random() * 0.2; // High NIR reflectance
            }
            reflectance.push(value);
        }

        this.charts.spectral = new Chart(ctx, {
            type: 'line',
            data: {
                labels: wavelengths,
                datasets: [{
                    label: 'Reflectance',
                    data: reflectance,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Wavelength (nm)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Reflectance'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    // AI Models Loading and Management
    async loadAIModels() {
        try {
            console.log('Loading AI models...');
            
            // Load TensorFlow.js models for crop disease detection
            this.aiModels.diseaseDetection = await this.loadDiseaseModel();
            this.aiModels.stressDetection = await this.loadStressModel();
            this.aiModels.yieldPrediction = await this.loadYieldModel();
            
            console.log('AI models loaded successfully');
        } catch (error) {
            console.error('Failed to load AI models:', error);
        }
    }

    async loadDiseaseModel() {
        // Simulate loading a CNN model for disease detection
        return {
            predict: (imageData) => {
                // Simulated disease prediction
                return {
                    confidence: 0.89,
                    disease: 'Powdery Mildew',
                    severity: 'Medium',
                    recommendation: 'Apply fungicide treatment within 24-48 hours'
                };
            }
        };
    }

    async loadStressModel() {
        // Simulate loading an LSTM model for stress detection
        return {
            predict: (timeSeriesData) => {
                return {
                    stressLevel: 'Low',
                    stressType: 'Water',
                    confidence: 0.76,
                    timeToAction: '2-3 days'
                };
            }
        };
    }

    async loadYieldModel() {
        // Simulate loading a yield prediction model
        return {
            predict: (fieldData) => {
                return {
                    yieldIncrease: 2.3,
                    confidence: 0.87,
                    factors: ['Improved irrigation', 'Early disease detection']
                };
            }
        };
    }

    // Data Processing and Analysis
    async processHyperspectralData(imageData) {
        try {
            // Simulate hyperspectral image processing
            const indices = this.calculateVegetationIndices(imageData);
            const anomalies = this.detectSpectralAnomalies(imageData);
            const healthMap = this.generateHealthMap(indices);

            return {
                indices,
                anomalies,
                healthMap,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Hyperspectral processing failed:', error);
            throw error;
        }
    }

    calculateVegetationIndices(imageData) {
        // Simulate calculation of vegetation indices
        return {
            ndvi: 0.78 + (Math.random() - 0.5) * 0.1,
            evi: 0.65 + (Math.random() - 0.5) * 0.1,
            savi: 0.72 + (Math.random() - 0.5) * 0.1,
            ndre: 0.34 + (Math.random() - 0.5) * 0.05
        };
    }

    detectSpectralAnomalies(imageData) {
        // Simulate anomaly detection
        const anomalies = [];
        
        if (Math.random() > 0.7) {
            anomalies.push({
                type: 'disease_signature',
                location: { lat: 40.7135, lng: -74.0055 },
                confidence: 0.82,
                description: 'Spectral signature consistent with fungal infection'
            });
        }
        
        if (Math.random() > 0.8) {
            anomalies.push({
                type: 'nutrient_deficiency',
                location: { lat: 40.7125, lng: -74.0065 },
                confidence: 0.75,
                description: 'Chlorophyll absorption patterns indicate nitrogen deficiency'
            });
        }

        return anomalies;
    }

    generateHealthMap(indices) {
        // Generate health map based on vegetation indices
        const healthScore = (indices.ndvi + indices.evi + indices.savi) / 3;
        
        let healthCategory;
        if (healthScore >= 0.8) healthCategory = 'excellent';
        else if (healthScore >= 0.65) healthCategory = 'good';
        else if (healthScore >= 0.5) healthCategory = 'fair';
        else healthCategory = 'poor';

        return {
            overallScore: healthScore,
            category: healthCategory,
            zones: this.generateZoneHealth()
        };
    }

    generateZoneHealth() {
        // Simulate zone-specific health data
        return [
            { zone: 1, health: 0.85, status: 'excellent' },
            { zone: 2, health: 0.72, status: 'good' },
            { zone: 3, health: 0.58, status: 'fair' },
            { zone: 4, health: 0.79, status: 'good' },
            { zone: 5, health: 0.91, status: 'excellent' }
        ];
    }

    // Environmental Data Integration
    async integrateEnvironmentalData() {
        try {
            const sensorData = await this.fetchSensorData();
            const weatherData = await this.fetchWeatherData();
            
            // Fuse sensor and spectral data for enhanced analysis
            const fusedData = this.fuseSensorSpectralData(sensorData, this.spectralData);
            
            // Update AI predictions with environmental context
            await this.updatePredictionsWithEnvironmentalData(fusedData);
            
            return fusedData;
        } catch (error) {
            console.error('Environmental data integration failed:', error);
        }
    }

    async fetchSensorData() {
        // Simulate fetching real sensor data
        return {
            soilMoisture: 72 + (Math.random() - 0.5) * 10,
            airTemperature: 24 + (Math.random() - 0.5) * 8,
            humidity: 65 + (Math.random() - 0.5) * 20,
            leafWetness: Math.random() * 100,
            soilPH: 6.8 + (Math.random() - 0.5) * 0.8,
            timestamp: new Date().toISOString()
        };
    }

    async fetchWeatherData() {
        // Simulate weather API integration
        return {
            temperature: 24,
            humidity: 65,
            precipitation: 0,
            windSpeed: 12,
            forecast: [
                { day: 1, temp: 26, humidity: 70, rain: 20 },
                { day: 2, temp: 28, humidity: 75, rain: 60 },
                { day: 3, temp: 25, humidity: 68, rain: 10 }
            ]
        };
    }

    fuseSensorSpectralData(sensorData, spectralData) {
        // Advanced data fusion algorithm
        return {
            ...sensorData,
            ...spectralData,
            fusionScore: this.calculateFusionScore(sensorData, spectralData),
            correlations: this.findDataCorrelations(sensorData, spectralData)
        };
    }

    calculateFusionScore(sensor, spectral) {
        // Calculate confidence score for fused data
        const factors = [
            sensor.soilMoisture / 100,
            spectral.indices?.ndvi || 0.7,
            1 - (Math.abs(sensor.airTemperature - 25) / 40)
        ];
        
        return factors.reduce((sum, factor) => sum + factor, 0) / factors.length;
    }

    findDataCorrelations(sensor, spectral) {
        return {
            moistureVsNDVI: this.calculateCorrelation(sensor.soilMoisture, spectral.indices?.ndvi),
            tempVsStress: this.calculateCorrelation(sensor.airTemperature, spectral.stressLevel || 0),
            humidityVsDisease: this.calculateCorrelation(sensor.humidity, spectral.diseaseRisk || 0)
        };
    }

    calculateCorrelation(x, y) {
        // Simplified correlation calculation
        return Math.random() * 0.8 + 0.1; // Simulate correlation between 0.1 and 0.9
    }

    async updatePredictionsWithEnvironmentalData(fusedData) {
        // Enhanced AI predictions using environmental context
        if (this.aiModels.diseaseDetection && fusedData.humidity > 80) {
            const prediction = this.aiModels.diseaseDetection.predict(fusedData);
            this.alertSystem.checkDiseaseAlert(prediction);
        }

        if (this.aiModels.stressDetection && fusedData.soilMoisture < 50) {
            const prediction = this.aiModels.stressDetection.predict(fusedData);
            this.alertSystem.checkStressAlert(prediction);
        }
    }

    // Real-time Data Streaming
    startDataStreaming() {
        // Simulate real-time data updates
        setInterval(() => {
            this.updateRealTimeData();
        }, 30000); // Update every 30 seconds

        setInterval(() => {
            this.updateCharts();
        }, 60000); // Update charts every minute
    }

    async updateRealTimeData() {
        try {
            // Fetch latest sensor readings
            const sensorData = await this.fetchSensorData();
            
            // Update UI with new data
            this.updateMetricCards(sensorData);
            
            // Check for alerts
            this.alertSystem.checkThresholds(sensorData);
            
        } catch (error) {
            console.error('Real-time data update failed:', error);
        }
    }

    updateMetricCards(data) {
        // Update soil moisture
        const moistureElement = document.querySelector('.metric-card .metric-value.normal');
        if (moistureElement) {
            moistureElement.textContent = `${Math.round(data.soilMoisture)}%`;
        }

        // Update other metrics based on data
        this.updateHealthScore(data);
    }

    updateHealthScore(data) {
        // Calculate overall health score
        const healthScore = this.calculateOverallHealth(data);
        const healthElement = document.querySelector('.metric-card .metric-value.good');
        if (healthElement) {
            healthElement.textContent = `${Math.round(healthScore)}%`;
        }
    }

    calculateOverallHealth(data) {
        // Combine multiple factors for health score
        const moistureFactor = Math.min(data.soilMoisture / 70, 1);
        const tempFactor = 1 - Math.abs(data.airTemperature - 25) / 25;
        const humidityFactor = Math.min(data.humidity / 70, 1);
        
        return (moistureFactor + tempFactor + humidityFactor) / 3 * 100;
    }

    updateCharts() {
        // Update health chart with new data
        this.addDataToChart(this.charts.health, {
            ndvi: 0.75 + (Math.random() - 0.5) * 0.1,
            evi: 0.65 + (Math.random() - 0.5) * 0.1
        });

        // Update environment chart
        this.addDataToChart(this.charts.environment, {
            moisture: 70 + (Math.random() - 0.5) * 10
        });
    }

    addDataToChart(chart, data) {
        if (!chart) return;

        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Add new data point
        chart.data.labels.push(currentTime);
        chart.data.datasets.forEach((dataset, index) => {
            const value = Object.values(data)[index] || Math.random();
            dataset.data.push(value);
        });

        // Keep only last 10 data points
        if (chart.data.labels.length > 10) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }

        chart.update('none');
    }

    // Event Listeners and UI Interactions
    setupEventListeners() {
        // Map layer controls
        document.querySelectorAll('.btn-control').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchMapLayer(e.target.dataset.layer);
                document.querySelectorAll('.btn-control').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });

        // Environment chart tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchEnvironmentTab(e.target.dataset.tab);
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });

        // Field selector
        document.getElementById('fieldSelect')?.addEventListener('change', (e) => {
            this.switchField(e.target.value);
        });

        // Analyze button
        document.querySelector('.btn-analyze')?.addEventListener('click', () => {
            this.runSpectralAnalysis();
        });

        // Modal controls
        this.setupModalControls();

        // Navigation
        this.setupNavigation();
    }

    switchMapLayer(layer) {
        // Switch between different map overlays
        console.log(`Switching to ${layer} layer`);
        
        // Clear existing overlays and add new ones based on layer type
        this.map.eachLayer(layer => {
            if (layer instanceof L.Polygon) {
                this.map.removeLayer(layer);
            }
        });

        // Add appropriate overlay
        switch(layer) {
            case 'health':
                this.addHealthZones();
                break;
            case 'moisture':
                this.addMoistureZones();
                break;
            case 'temp':
                this.addTemperatureZones();
                break;
            case 'ndvi':
                this.addNDVIZones();
                break;
        }
    }

    addMoistureZones() {
        const moistureZones = [
            { coords: [[40.7145, -74.0075], [40.7145, -74.0055], [40.7130, -74.0055], [40.7130, -74.0075]], moisture: 'high' },
            { coords: [[40.7130, -74.0075], [40.7130, -74.0055], [40.7115, -74.0055], [40.7115, -74.0075]], moisture: 'medium' },
            { coords: [[40.7145, -74.0055], [40.7145, -74.0045], [40.7115, -74.0045], [40.7115, -74.0055]], moisture: 'low' }
        ];

        moistureZones.forEach((zone, index) => {
            const color = this.getMoistureColor(zone.moisture);
            L.polygon(zone.coords, {
                color: color,
                weight: 1,
                fillColor: color,
                fillOpacity: 0.6
            }).addTo(this.map).bindPopup(`Zone ${index + 1}: ${zone.moisture.toUpperCase()} moisture`);
        });
    }

    addTemperatureZones() {
        const tempZones = [
            { coords: [[40.7145, -74.0075], [40.7145, -74.0055], [40.7130, -74.0055], [40.7130, -74.0075]], temp: 'optimal' },
            { coords: [[40.7130, -74.0075], [40.7130, -74.0055], [40.7115, -74.0055], [40.7115, -74.0075]], temp: 'warm' },
            { coords: [[40.7145, -74.0055], [40.7145, -74.0045], [40.7115, -74.0045], [40.7115, -74.0055]], temp: 'hot' }
        ];

        tempZones.forEach((zone, index) => {
            const color = this.getTemperatureColor(zone.temp);
            L.polygon(zone.coords, {
                color: color,
                weight: 1,
                fillColor: color,
                fillOpacity: 0.6
            }).addTo(this.map).bindPopup(`Zone ${index + 1}: ${zone.temp.toUpperCase()} temperature`);
        });
    }

    addNDVIZones() {
        const ndviZones = [
            { coords: [[40.7145, -74.0075], [40.7145, -74.0055], [40.7130, -74.0055], [40.7130, -74.0075]], ndvi: 0.85 },
            { coords: [[40.7130, -74.0075], [40.7130, -74.0055], [40.7115, -74.0055], [40.7115, -74.0075]], ndvi: 0.72 },
            { coords: [[40.7145, -74.0055], [40.7145, -74.0045], [40.7115, -74.0045], [40.7115, -74.0055]], ndvi: 0.58 }
        ];

        ndviZones.forEach((zone, index) => {
            const color = this.getNDVIColor(zone.ndvi);
            L.polygon(zone.coords, {
                color: color,
                weight: 1,
                fillColor: color,
                fillOpacity: 0.6
            }).addTo(this.map).bindPopup(`Zone ${index + 1}: NDVI ${zone.ndvi}`);
        });
    }

    getMoistureColor(moisture) {
        const colors = { high: '#0ea5e9', medium: '#22d3ee', low: '#fbbf24' };
        return colors[moisture] || '#6b7280';
    }

    getTemperatureColor(temp) {
        const colors = { optimal: '#10b981', warm: '#f59e0b', hot: '#ef4444' };
        return colors[temp] || '#6b7280';
    }

    getNDVIColor(ndvi) {
        if (ndvi >= 0.8) return '#059669';
        if (ndvi >= 0.65) return '#84cc16';
        if (ndvi >= 0.5) return '#f59e0b';
        return '#ef4444';
    }

    switchEnvironmentTab(tab) {
        // Switch environment chart data
        const chart = this.charts.environment;
        if (!chart) return;

        let newData, label;
        switch(tab) {
            case 'moisture':
                newData = [72, 71, 69, 65, 68, 70, 72];
                label = 'Soil Moisture (%)';
                break;
            case 'temp':
                newData = [22, 18, 25, 28, 26, 23, 20];
                label = 'Temperature (°C)';
                break;
            case 'humidity':
                newData = [65, 68, 72, 78, 75, 70, 67];
                label = 'Humidity (%)';
                break;
        }

        chart.data.datasets[0].data = newData;
        chart.data.datasets[0].label = label;
        chart.update();
    }

    switchField(fieldId) {
        console.log(`Switching to field: ${fieldId}`);
        // Update map view and data for selected field
        // This would typically involve API calls to fetch field-specific data
    }

    async runSpectralAnalysis() {
        const button = document.querySelector('.btn-analyze');
        button.classList.add('loading');
        button.textContent = 'Analyzing...';

        try {
            // Simulate spectral analysis
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Update spectral chart with new analysis
            this.updateSpectralAnalysis();
            
            // Update vegetation indices
            this.updateVegetationIndices();
            
            // Update AI insights
            this.updateAIInsights();

        } catch (error) {
            console.error('Spectral analysis failed:', error);
        } finally {
            button.classList.remove('loading');
            button.innerHTML = '<i class="fas fa-microscope"></i> Run Analysis';
        }
    }

    updateSpectralAnalysis() {
        // Update spectral signature chart with new data
        const chart = this.charts.spectral;
        if (chart) {
            // Generate new spectral signature
            const newData = chart.data.datasets[0].data.map(value => 
                value + (Math.random() - 0.5) * 0.1
            );
            chart.data.datasets[0].data = newData;
            chart.update();
        }
    }

    updateVegetationIndices() {
        // Update vegetation index values
        const indices = this.calculateVegetationIndices();
        
        document.querySelectorAll('.index-item').forEach((item, index) => {
            const valueElement = item.querySelector('.index-value');
            const values = Object.values(indices);
            if (valueElement && values[index]) {
                valueElement.textContent = values[index].toFixed(2);
            }
        });
    }

    updateAIInsights() {
        // Update AI insights with new analysis
        const insights = [
            'Crop stress reduced by 15% in zones 3-5 after irrigation',
            'Chlorophyll content improved by 8% in northern section',
            'Disease risk decreased to low following fungicide application'
        ];

        const insightElements = document.querySelectorAll('.insight-item span');
        insightElements.forEach((element, index) => {
            if (insights[index]) {
                element.textContent = insights[index];
            }
        });
    }

    setupModalControls() {
        const modal = document.getElementById('alertModal');
        const closeBtn = document.querySelector('.close');

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                modal.style.display = 'none';
            });
        }

        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }

    setupNavigation() {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
                
                // Handle navigation logic here
                const section = e.target.getAttribute('href').substring(1);
                this.navigateToSection(section);
            });
        });
    }

    navigateToSection(section) {
        console.log(`Navigating to: ${section}`);
        // Implement section navigation logic
    }

    async loadInitialData() {
        try {
            // Load initial field data
            await this.integrateEnvironmentalData();
            
            // Update vegetation indices
            this.updateVegetationIndices();
            
            // Generate initial predictions
            this.generateInitialPredictions();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }

    generateInitialPredictions() {
        // Generate initial AI predictions for dashboard
        const predictions = {
            yield: { increase: 2.3, confidence: 87 },
            disease: { risk: 'Medium', type: 'Powdery Mildew', timeline: '3-4 days' },
            irrigation: { action: 'Increase by 15%', zones: '2, 4, 7', timing: '24-48 hours' }
        };

        this.updatePredictionCards(predictions);
    }

    updatePredictionCards(predictions) {
        // Update yield forecast
        const yieldValue = document.querySelector('.forecast-value');
        if (yieldValue) {
            yieldValue.textContent = `+${predictions.yield.increase}%`;
        }

        // Update disease risk
        const riskLevel = document.querySelector('.risk-level');
        if (riskLevel) {
            riskLevel.textContent = `${predictions.disease.risk} Risk`;
            riskLevel.className = `risk-level ${predictions.disease.risk.toLowerCase()}`;
        }

        // Update irrigation recommendation
        const irrigationAction = document.querySelector('.irrigation-action');
        if (irrigationAction) {
            irrigationAction.textContent = predictions.irrigation.action;
        }
    }

    showError(message) {
        // Display error message to user
        console.error(message);
        // Could implement toast notifications here
    }
}

// Alert System Class
class AlertSystem {
    constructor() {
        this.thresholds = {
            soilMoisture: { min: 40, max: 90 },
            temperature: { min: 15, max: 35 },
            humidity: { min: 30, max: 85 },
            diseaseRisk: 0.7,
            stressLevel: 0.6
        };
        this.alerts = [];
    }

    checkThresholds(data) {
        this.checkSoilMoisture(data.soilMoisture);
        this.checkTemperature(data.airTemperature);
        this.checkHumidity(data.humidity);
    }

    checkSoilMoisture(moisture) {
        if (moisture < this.thresholds.soilMoisture.min) {
            this.createAlert('low_moisture', 'Critical', 'Soil moisture critically low - immediate irrigation required');
        }
    }

    checkTemperature(temperature) {
        if (temperature > this.thresholds.temperature.max) {
            this.createAlert('high_temp', 'High', 'Temperature stress detected - consider additional cooling measures');
        }
    }

    checkHumidity(humidity) {
        if (humidity > this.thresholds.humidity.max) {
            this.createAlert('high_humidity', 'Medium', 'High humidity levels increase disease risk');
        }
    }

    checkDiseaseAlert(prediction) {
        if (prediction.confidence > this.thresholds.diseaseRisk) {
            this.createAlert('disease_risk', 'High', `${prediction.disease} risk detected - ${prediction.recommendation}`);
        }
    }

    checkStressAlert(prediction) {
        if (prediction.confidence > this.thresholds.stressLevel) {
            this.createAlert('stress_detected', 'Medium', `${prediction.stressType} stress detected - action needed in ${prediction.timeToAction}`);
        }
    }

    createAlert(type, priority, message) {
        const alert = {
            id: Date.now(),
            type,
            priority,
            message,
            timestamp: new Date().toISOString(),
            acknowledged: false
        };

        this.alerts.unshift(alert);
        this.displayAlert(alert);
        this.updateNotificationBadge();
    }

    displayAlert(alert) {
        if (alert.priority === 'Critical' || alert.priority === 'High') {
            this.showModalAlert(alert);
        }
        
        this.addToAlertList(alert);
    }

    showModalAlert(alert) {
        const modal = document.getElementById('alertModal');
        const title = document.getElementById('alertTitle');
        const description = document.getElementById('alertDescription');

        if (title) title.textContent = alert.message.split(' - ')[0];
        if (description) description.textContent = alert.message;
        if (modal) modal.style.display = 'block';
    }

    addToAlertList(alert) {
        const alertList = document.querySelector('.alert-list');
        if (!alertList) return;

        const alertElement = document.createElement('div');
        alertElement.className = `alert-item ${alert.priority.toLowerCase()}`;
        alertElement.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <div>
                <div class="alert-title">${alert.message.split(' - ')[0]}</div>
                <div class="alert-time">Just now</div>
            </div>
        `;

        alertList.insertBefore(alertElement, alertList.firstChild);

        // Keep only last 5 alerts visible
        const alerts = alertList.querySelectorAll('.alert-item');
        if (alerts.length > 5) {
            alertList.removeChild(alerts[alerts.length - 1]);
        }
    }

    updateNotificationBadge() {
        const badge = document.querySelector('.notification-badge');
        if (badge) {
            const unacknowledged = this.alerts.filter(alert => !alert.acknowledged).length;
            badge.textContent = unacknowledged;
            badge.style.display = unacknowledged > 0 ? 'block' : 'none';
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing Agricultural AI Platform...');
    new AgriculturePlatform();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AgriculturePlatform, AlertSystem };
}
