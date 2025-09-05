const winston = require('winston');

class SensorService {
    constructor() {
        this.logger = winston.createLogger({
            level: 'info',
            format: winston.format.simple(),
            transports: [new winston.transports.Console()]
        });
        
        this.sensorTypes = {
            soilMoisture: { min: 0, max: 100, unit: '%', normalRange: [40, 80] },
            temperature: { min: -10, max: 50, unit: '°C', normalRange: [15, 35] },
            humidity: { min: 0, max: 100, unit: '%', normalRange: [30, 80] },
            leafWetness: { min: 0, max: 24, unit: 'hours', normalRange: [0, 6] },
            soilPH: { min: 4, max: 9, unit: 'pH', normalRange: [6, 7.5] },
            lightIntensity: { min: 0, max: 100000, unit: 'lux', normalRange: [20000, 80000] }
        };
    }

    async collectAllSensorData() {
        try {
            this.logger.info('Collecting sensor data from all fields...');
            
            const fields = await this.getActiveFields();
            const collectionPromises = fields.map(field => this.collectFieldSensorData(field._id));
            
            const results = await Promise.all(collectionPromises);
            
            this.logger.info(`Collected sensor data from ${results.length} fields`);
            return results;
        } catch (error) {
            this.logger.error('Sensor data collection failed:', error);
            throw error;
        }
    }

    async collectFieldSensorData(fieldId) {
        try {
            const sensors = await this.getFieldSensors(fieldId);
            const sensorDataPromises = sensors.map(sensor => this.readSensorData(sensor));
            
            const sensorReadings = await Promise.all(sensorDataPromises);
            
            const aggregatedData = this.aggregateSensorData(sensorReadings);
            
            // Store in database
            await this.storeSensorData(fieldId, aggregatedData);
            
            // Check for alerts
            this.checkSensorAlerts(fieldId, aggregatedData);
            
            return {
                fieldId,
                data: aggregatedData,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.logger.error(`Sensor collection failed for field ${fieldId}:`, error);
            throw error;
        }
    }

    async getFieldSensors(fieldId) {
        // Simulate getting sensors for a field
        return [
            { id: `${fieldId}_soil_1`, type: 'soilMoisture', location: { lat: 40.7140, lng: -74.0070 } },
            { id: `${fieldId}_soil_2`, type: 'soilMoisture', location: { lat: 40.7125, lng: -74.0060 } },
            { id: `${fieldId}_weather_1`, type: 'temperature', location: { lat: 40.7135, lng: -74.0065 } },
            { id: `${fieldId}_weather_1`, type: 'humidity', location: { lat: 40.7135, lng: -74.0065 } },
            { id: `${fieldId}_leaf_1`, type: 'leafWetness', location: { lat: 40.7130, lng: -74.0055 } },
            { id: `${fieldId}_soil_ph_1`, type: 'soilPH', location: { lat: 40.7140, lng: -74.0070 } }
        ];
    }

    async readSensorData(sensor) {
        // Simulate reading data from actual sensors
        const sensorConfig = this.sensorTypes[sensor.type];
        if (!sensorConfig) {
            throw new Error(`Unknown sensor type: ${sensor.type}`);
        }

        // Generate realistic sensor readings with some noise
        const baseValue = (sensorConfig.normalRange[0] + sensorConfig.normalRange[1]) / 2;
        const range = sensorConfig.normalRange[1] - sensorConfig.normalRange[0];
        const noise = (Math.random() - 0.5) * range * 0.3; // ±30% of normal range
        
        let value = baseValue + noise;
        
        // Add some realistic variations based on time of day for certain sensors
        if (sensor.type === 'temperature') {
            const hour = new Date().getHours();
            const tempVariation = Math.sin((hour - 6) * Math.PI / 12) * 8; // Peak at 2 PM
            value += tempVariation;
        } else if (sensor.type === 'humidity') {
            const hour = new Date().getHours();
            const humidityVariation = -Math.sin((hour - 6) * Math.PI / 12) * 15; // Inverse of temperature
            value += humidityVariation;
        }
        
        // Clamp to sensor limits
        value = Math.max(sensorConfig.min, Math.min(sensorConfig.max, value));
        
        return {
            sensorId: sensor.id,
            type: sensor.type,
            value: Math.round(value * 100) / 100, // Round to 2 decimal places
            unit: sensorConfig.unit,
            location: sensor.location,
            timestamp: new Date().toISOString(),
            quality: this.assessDataQuality(value, sensorConfig),
            batteryLevel: 80 + Math.random() * 20 // Simulate battery level
        };
    }

    assessDataQuality(value, config) {
        // Assess data quality based on sensor specs and normal ranges
        const [normalMin, normalMax] = config.normalRange;
        
        if (value >= normalMin && value <= normalMax) {
            return 'good';
        } else if (value >= config.min && value <= config.max) {
            return 'acceptable';
        } else {
            return 'poor';
        }
    }

    aggregateSensorData(sensorReadings) {
        const aggregated = {};
        
        // Group readings by sensor type
        const groupedReadings = {};
        sensorReadings.forEach(reading => {
            if (!groupedReadings[reading.type]) {
                groupedReadings[reading.type] = [];
            }
            groupedReadings[reading.type].push(reading);
        });
        
        // Calculate statistics for each sensor type
        Object.keys(groupedReadings).forEach(type => {
            const readings = groupedReadings[type];
            const values = readings.map(r => r.value);
            
            aggregated[type] = {
                current: values[values.length - 1], // Most recent value
                average: this.calculateMean(values),
                min: Math.min(...values),
                max: Math.max(...values),
                trend: this.calculateTrend(values),
                quality: this.aggregateQuality(readings),
                sensorCount: readings.length,
                unit: readings[0].unit
            };
        });
        
        return aggregated;
    }

    calculateMean(values) {
        if (values.length === 0) return 0;
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }

    calculateTrend(values) {
        if (values.length < 2) return 'stable';
        
        const recent = values.slice(-3); // Last 3 readings
        const older = values.slice(-6, -3); // Previous 3 readings
        
        if (recent.length === 0 || older.length === 0) return 'stable';
        
        const recentAvg = this.calculateMean(recent);
        const olderAvg = this.calculateMean(older);
        
        const change = ((recentAvg - olderAvg) / olderAvg) * 100;
        
        if (change > 5) return 'increasing';
        if (change < -5) return 'decreasing';
        return 'stable';
    }

    aggregateQuality(readings) {
        const qualityScores = { good: 3, acceptable: 2, poor: 1 };
        const totalScore = readings.reduce((sum, reading) => sum + qualityScores[reading.quality], 0);
        const avgScore = totalScore / readings.length;
        
        if (avgScore >= 2.5) return 'good';
        if (avgScore >= 1.5) return 'acceptable';
        return 'poor';
    }

    checkSensorAlerts(fieldId, aggregatedData) {
        const alerts = [];
        
        Object.keys(aggregatedData).forEach(sensorType => {
            const data = aggregatedData[sensorType];
            const config = this.sensorTypes[sensorType];
            const [normalMin, normalMax] = config.normalRange;
            
            if (data.current < normalMin) {
                alerts.push({
                    fieldId,
                    type: 'low_reading',
                    sensorType,
                    message: `${sensorType} is below normal range: ${data.current}${data.unit}`,
                    severity: data.current < normalMin * 0.8 ? 'high' : 'medium',
                    value: data.current,
                    threshold: normalMin
                });
            } else if (data.current > normalMax) {
                alerts.push({
                    fieldId,
                    type: 'high_reading',
                    sensorType,
                    message: `${sensorType} is above normal range: ${data.current}${data.unit}`,
                    severity: data.current > normalMax * 1.2 ? 'high' : 'medium',
                    value: data.current,
                    threshold: normalMax
                });
            }
            
            if (data.quality === 'poor') {
                alerts.push({
                    fieldId,
                    type: 'data_quality',
                    sensorType,
                    message: `Poor data quality detected for ${sensorType} sensors`,
                    severity: 'low'
                });
            }
        });
        
        // Emit alerts if any
        if (alerts.length > 0) {
            this.emitSensorAlerts(alerts);
        }
    }

    emitSensorAlerts(alerts) {
        // This would typically integrate with the alert service
        alerts.forEach(alert => {
            this.logger.warn(`Sensor Alert: ${alert.message}`);
        });
    }

    async getLatestSensorData(fieldId) {
        try {
            // Simulate retrieving latest sensor data from database
            const latestData = await this.collectFieldSensorData(fieldId);
            return latestData;
        } catch (error) {
            this.logger.error(`Failed to get latest sensor data for field ${fieldId}:`, error);
            throw error;
        }
    }

    async storeSensorData(fieldId, data) {
        // Simulate storing data in database
        this.logger.info(`Storing sensor data for field ${fieldId}`);
        // In production, this would save to MongoDB or other database
    }

    async getActiveFields() {
        // Simulate getting active fields from database
        return [
            { _id: 'field1', name: 'North Field' },
            { _id: 'field2', name: 'South Field' },
            { _id: 'field3', name: 'East Field' }
        ];
    }

    // Historical data analysis
    async getHistoricalTrends(fieldId, sensorType, timeRange) {
        try {
            this.logger.info(`Getting historical trends for ${sensorType} in field ${fieldId}`);
            
            // Simulate historical data retrieval
            const historicalData = this.generateHistoricalData(sensorType, timeRange);
            
            const trends = {
                data: historicalData,
                statistics: this.calculateHistoricalStatistics(historicalData),
                anomalies: this.detectHistoricalAnomalies(historicalData),
                predictions: this.generateShortTermForecast(historicalData)
            };
            
            return trends;
        } catch (error) {
            this.logger.error('Historical trends analysis failed:', error);
            throw error;
        }
    }

    generateHistoricalData(sensorType, timeRange) {
        const config = this.sensorTypes[sensorType];
        const data = [];
        const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
        
        for (let i = days; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            
            // Generate realistic historical values
            const baseValue = (config.normalRange[0] + config.normalRange[1]) / 2;
            const seasonalVariation = Math.sin((date.getMonth() * 2 * Math.PI) / 12) * 5;
            const randomVariation = (Math.random() - 0.5) * 10;
            
            const value = baseValue + seasonalVariation + randomVariation;
            
            data.push({
                timestamp: date.toISOString(),
                value: Math.max(config.min, Math.min(config.max, value)),
                quality: Math.random() > 0.1 ? 'good' : 'acceptable'
            });
        }
        
        return data;
    }

    calculateHistoricalStatistics(data) {
        const values = data.map(d => d.value);
        
        return {
            mean: this.calculateMean(values),
            median: this.calculateMedian(values),
            min: Math.min(...values),
            max: Math.max(...values),
            standardDeviation: this.calculateStandardDeviation(values),
            trend: this.calculateLongTermTrend(values)
        };
    }

    calculateMedian(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        
        return sorted.length % 2 === 0 
            ? (sorted[mid - 1] + sorted[mid]) / 2 
            : sorted[mid];
    }

    calculateStandardDeviation(values) {
        const mean = this.calculateMean(values);
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }

    calculateLongTermTrend(values) {
        // Simple linear regression for long-term trend
        const n = values.length;
        const x = Array.from({ length: n }, (_, i) => i);
        
        const sumX = x.reduce((sum, val) => sum + val, 0);
        const sumY = values.reduce((sum, val) => sum + val, 0);
        const sumXY = x.reduce((sum, val, i) => sum + val * values[i], 0);
        const sumXX = x.reduce((sum, val) => sum + val * val, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        
        if (slope > 0.1) return 'increasing';
        if (slope < -0.1) return 'decreasing';
        return 'stable';
    }

    detectHistoricalAnomalies(data) {
        const values = data.map(d => d.value);
        const mean = this.calculateMean(values);
        const std = this.calculateStandardDeviation(values);
        
        const anomalies = [];
        
        data.forEach((point, index) => {
            const zScore = Math.abs((point.value - mean) / std);
            
            if (zScore > 2.5) { // 2.5 sigma threshold
                anomalies.push({
                    timestamp: point.timestamp,
                    value: point.value,
                    zScore,
                    severity: zScore > 3 ? 'high' : 'medium'
                });
            }
        });
        
        return anomalies;
    }

    generateShortTermForecast(data) {
        // Simple time series forecasting using moving average
        const windowSize = Math.min(7, data.length);
        const recentValues = data.slice(-windowSize).map(d => d.value);
        const trend = this.calculateTrend(recentValues);
        
        const forecast = [];
        let lastValue = recentValues[recentValues.length - 1];
        
        for (let i = 1; i <= 3; i++) { // 3-day forecast
            const forecastDate = new Date();
            forecastDate.setDate(forecastDate.getDate() + i);
            
            // Simple trend-based prediction
            let change = 0;
            if (trend === 'increasing') change = Math.random() * 2;
            if (trend === 'decreasing') change = -Math.random() * 2;
            
            const forecastValue = lastValue + change + (Math.random() - 0.5);
            
            forecast.push({
                timestamp: forecastDate.toISOString(),
                predictedValue: Math.round(forecastValue * 100) / 100,
                confidence: 0.7 - (i * 0.1) // Decreasing confidence over time
            });
            
            lastValue = forecastValue;
        }
        
        return forecast;
    }

    // Calibration and maintenance
    async calibrateSensor(sensorId, calibrationData) {
        try {
            this.logger.info(`Calibrating sensor ${sensorId}`);
            
            // Simulate sensor calibration process
            const calibrationResult = {
                sensorId,
                previousCalibration: calibrationData.previousCalibration,
                newCalibration: calibrationData.newCalibration,
                offset: calibrationData.newCalibration - calibrationData.previousCalibration,
                calibratedAt: new Date().toISOString(),
                calibratedBy: calibrationData.technician,
                success: Math.abs(calibrationData.newCalibration - calibrationData.expected) < 0.1
            };
            
            // Store calibration record
            await this.storeCalibrationRecord(calibrationResult);
            
            return calibrationResult;
        } catch (error) {
            this.logger.error(`Sensor calibration failed for ${sensorId}:`, error);
            throw error;
        }
    }

    async storeCalibrationRecord(calibrationResult) {
        this.logger.info(`Storing calibration record for sensor ${calibrationResult.sensorId}`);
        // In production, store in database
    }

    async getSensorHealth(fieldId) {
        try {
            const sensors = await this.getFieldSensors(fieldId);
            const healthData = [];
            
            for (const sensor of sensors) {
                const reading = await this.readSensorData(sensor);
                const health = {
                    sensorId: sensor.id,
                    type: sensor.type,
                    status: this.determineSensorStatus(reading),
                    batteryLevel: reading.batteryLevel,
                    dataQuality: reading.quality,
                    lastReading: reading.timestamp,
                    location: sensor.location
                };
                
                healthData.push(health);
            }
            
            return {
                fieldId,
                sensors: healthData,
                overallHealth: this.calculateOverallSensorHealth(healthData),
                maintenanceNeeded: healthData.filter(s => s.status !== 'operational').length
            };
        } catch (error) {
            this.logger.error(`Failed to get sensor health for field ${fieldId}:`, error);
            throw error;
        }
    }

    determineSensorStatus(reading) {
        if (reading.batteryLevel < 20) return 'low_battery';
        if (reading.quality === 'poor') return 'needs_calibration';
        if (reading.batteryLevel < 50 && reading.quality === 'acceptable') return 'maintenance_soon';
        return 'operational';
    }

    calculateOverallSensorHealth(healthData) {
        const operationalCount = healthData.filter(s => s.status === 'operational').length;
        const healthPercentage = (operationalCount / healthData.length) * 100;
        
        if (healthPercentage >= 90) return 'excellent';
        if (healthPercentage >= 75) return 'good';
        if (healthPercentage >= 60) return 'fair';
        return 'poor';
    }
}

module.exports = SensorService;
