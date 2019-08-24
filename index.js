/* Copyright 2019 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
tf.setBackend('cpu');

function random_uniform(min, max) {
    return Math.random() * (max - min) + min;
}

function centerCoordinates(xy) {
    return [(xy[0] - 0.5 * width) * 14.0/width, (xy[1] - 0.5 * height) * 14.0/height]
}

width = 256;
height = 256;
radius = 4;
num_samples = 256

function sigmoid(x) {
    var val = 1.0/(1.0+Math.pow(Math.E, -x));
    return Number(val.toFixed(2))
}

function getTemperature1() {
  return d3.select("#t1").property("value") / 100.0
}

function getTemperature2() {
  return d3.select("#t2").property("value") / 100.0
}

function disableNoiseLevel() {
    d3.select("#noise-level").property("disabled", true);
    d3.select(".slider-group.noise-level").style("visibility", "hidden");
}

function enableNoiseLevel() {
    d3.select("#noise-level").property("disabled", false);
    d3.select(".slider-group.noise-level").style("visibility", "visible");
}

function getNoiseLevel() {
    return d3.select("#noise-level").property("value") / 100.0
}

function shouldResetModel() {
  return d3.select("#reset-weights").property("checked");
}
disableNoiseLevel()

function updateStatus(statusMessage) {
    d3.select("#status").text(statusMessage);
}

d3.select("#noise-level").on("input", function() {
    updateNoiseLevel(+this.value);
});

function updateNoiseLevel(noiseLevel) {
    d3.select("#noise-value").text(noiseLevel + "%");
    d3.select("#noise-level").property("value", noiseLevel);
}

d3.select("#t1").on("input", function() {
    updateT1(+this.value);
});

d3.select("#t1").on("change", function() {
    resetModel()
    updateDataset();
    renderLossPlot();
});

function updateT1(t1Value) {
    d3.select("#t1-value").text(t1Value / 100.0);
    d3.select("#t1").property("value", t1Value);
}

d3.select("#t2").on("input", function() {
    updateT2(+this.value);
});

d3.select("#t2").on("change", function() {
    resetModel()
    updateDataset();
    renderLossPlot();
});

function updateT2(t2Value) {
    d3.select("#t2-value").text(t2Value / 100.0);
    d3.select("#t2").property("value", t2Value);
}

d3.select("#noise-level").on("change", function() {
    var svg = d3.select("#svg-canvas")
    if (d3.select("#low-noise").property("checked")) {
        generateLowMarginNoise(svg)
    } else if (d3.select("#high-noise").property("checked")) {
        generateHighMarginNoise(svg)
    } else if (d3.select("#random-noise").property("checked")) {
        generateRandomNoise(svg)
    }

});

function isNoisyPoint(i, currentLabel) {
    if (i % 2 == 0) {
        return currentLabel == 0.0;
    } else {
        return currentLabel == 1.0;
    }
}

const outerRingRadiusStart = 0.33
const outerRingRadiusEnd = 0.48
const innerRingRadiusEnd = 0.2
const deltaRadius = 0.03
var model = createFeedForwardModel()
const numTrainingPoints = 1000;

var trainingDataPoints = d3.range(numTrainingPoints).map(function(d) {
    if (d % 2 == 0) {
        var radius = random_uniform(width * outerRingRadiusStart, width * outerRingRadiusEnd)
        var angle = random_uniform(0, 2 * 3.147)
        return {
            x: Math.round(width * 0.5 + radius * Math.cos(angle)),
            y: Math.round(height * 0.5 + radius * Math.sin(angle)),
            radius: radius,
            label: 1.0,
            color: "#ff8b6a",
        };
    } else {
        var radius = random_uniform(0, width * innerRingRadiusEnd)
        var angle = random_uniform(0, 2 * 3.147)
        return {

            x: Math.round(width * 0.5 + radius * Math.cos(angle)),
            y: Math.round(height * 0.5 + radius * Math.sin(angle)),
            radius: radius,
            label: 0.0,
            color: "#248ea9",
        };
    }
});

function renderTrainingData(svg, trainingDataPointsArg) {
    var goodDataPoints = []
    var noisyDataPoints = []
    for (var i = 0; i < trainingDataPointsArg.length; ++i) {
        if (isNoisyPoint(i, trainingDataPointsArg[i].label)) {
            noisyDataPoints.push(trainingDataPointsArg[i])
        } else {
            goodDataPoints.push(trainingDataPointsArg[i])
        }
    }
    var reorderedTrainingDataPoints = goodDataPoints.concat(noisyDataPoints)

    svg.selectAll("circle").remove();

    var td = svg.selectAll("circle")
        .data(reorderedTrainingDataPoints)

    td.enter().append("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", radius)
        .attr("class", "trainingData")
        .style("fill", function(d) {
            return d.color;
        })
        .call(d3.drag().on("start", dragStarted).on("drag", duringDragging).on("end", dragEnd));

    td.attr("class", "trainingData")
        .style("fill", function(d) {
            return d.color;
        });
    td.exit().remove();
}

function generateCleanDataPoints(svg) {
    disableNoiseLevel()
    for (var i = 0; i < trainingDataPoints.length; ++i) {
        if (i % 2 == 0) {
            trainingDataPoints[i].label = 1.0;
            trainingDataPoints[i].color = "#ff8b6a";
        } else {
            trainingDataPoints[i].label = 0.0;
            trainingDataPoints[i].color = "#248ea9";
        }
    }
    renderTrainingData(svg, trainingDataPoints);
    resetModel()
}

function generateLowMarginNoise(svg) {
    enableNoiseLevel()
    const noiseLevel = getNoiseLevel()
    for (var i = 0; i < trainingDataPoints.length; ++i) {
        if (i % 2 == 0) {
            trainingDataPoints[i].label = 1.0;
            trainingDataPoints[i].color = "#ff8b6a";
            if (trainingDataPoints[i].radius < (outerRingRadiusStart + deltaRadius)* width) {
                var rand = random_uniform(0, 1.0);
                if (rand < noiseLevel) {
                    trainingDataPoints[i].label = 0.0;
                    trainingDataPoints[i].color = "#248ea9";
                }
            }
        } else {
            trainingDataPoints[i].label = 0.0;
            trainingDataPoints[i].color = "#248ea9";

            if (trainingDataPoints[i].radius > (innerRingRadiusEnd - deltaRadius) * width) {
                var rand = random_uniform(0, 1.0);
                if (rand < noiseLevel) {
                    trainingDataPoints[i].label = 1.0;
                    trainingDataPoints[i].color = "#ff8b6a";
                }
            }
        }
    }
    renderTrainingData(svg, trainingDataPoints);
    resetModel()
}

function generateHighMarginNoise(svg) {
    enableNoiseLevel()
    const noiseLevel = getNoiseLevel()
    for (var i = 0; i < trainingDataPoints.length; ++i) {
        if (i % 2 == 0) {
            trainingDataPoints[i].label = 1.0;
            trainingDataPoints[i].color = "#ff8b6a";
            if (trainingDataPoints[i].radius > (outerRingRadiusEnd - deltaRadius) * width) {
                var rand = random_uniform(0, 1.0);
                if (rand < noiseLevel) {
                    trainingDataPoints[i].label = 0.0;
                    trainingDataPoints[i].color = "#248ea9";
                }
            }
        } else {
            trainingDataPoints[i].label = 0.0;
            trainingDataPoints[i].color = "#248ea9";

            if (trainingDataPoints[i].radius < (deltaRadius) * width) {
                var rand = random_uniform(0, 1.0);
                if (rand < noiseLevel) {
                    trainingDataPoints[i].label = 1.0;
                    trainingDataPoints[i].color = "#ff8b6a";
                }
            }
        }
    }
    renderTrainingData(svg, trainingDataPoints);
    resetModel()
}

function generateRandomNoise(svg) {
    enableNoiseLevel()
    const noiseLevel = getNoiseLevel()
    for (var i = 0; i < trainingDataPoints.length; ++i) {
        if (i % 2 == 0) {
            trainingDataPoints[i].label = 1.0;
            trainingDataPoints[i].color = "#ff8b6a";
            var rand = random_uniform(0, 1.0);
            if (rand < noiseLevel) {
                trainingDataPoints[i].label = 0.0;
                trainingDataPoints[i].color = "#248ea9";
            }
        } else {
            trainingDataPoints[i].label = 0.0;
            trainingDataPoints[i].color = "#248ea9";
            var rand = random_uniform(0, 1.0);
            if (rand < noiseLevel) {
                trainingDataPoints[i].label = 1.0;
                trainingDataPoints[i].color = "#ff8b6a";
            }
        }
    }
    renderTrainingData(svg, trainingDataPoints);
    resetModel()
}

function render() {
    var svg = d3.select("#svg-canvas")
        .attr("height", height)
        .attr("width", width);

    var canvas = d3.select("#decision-surface-canvas")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height)
        .style("width", width + "px")
        .style("height", height + "px")

    renderTrainingData(svg, trainingDataPoints);

    d3.select("#no-noise").on("click", function() {
      generateCleanDataPoints(svg)
    });
    d3.select("#low-noise").on("click", function() {
      generateLowMarginNoise(svg)
    });
    d3.select("#high-noise").on("click", function() {
      generateHighMarginNoise(svg)
    });
    d3.select("#random-noise").on("click", function() {
      generateRandomNoise(svg)
    });
    return canvas;
}

canvas = render()
train()

d3.select("#svg-canvas").on('mousemove', function(){
  var mouseX = d3.event.layerX || d3.event.offsetX;
  var mouseY = d3.event.layerY || d3.event.offsety;
  current_xy = tf.tensor(centerCoordinates([mouseX, mouseY]), [1, 2])
  activation = parseFloat(model.predict(current_xy).arraySync()[0])
  t2 = getTemperature2()
  activation_t = tf.tensor([activation], [1])
  pLabel = parseFloat(tf.split(temperedSigmoid(activation_t, t2, 5), 2, 1)[1].arraySync()[0])

  d3.select('#tooltip')
    .style('opacity', 0.8)
    .style('top', d3.event.pageY + 5 + 'px')
    .style('left', d3.event.pageX + 5 + 'px')
    .html('activation: ' + Number(activation.toFixed(2)) + ' probability: ' + Number(pLabel.toFixed(4)));
})

d3.select("#svg-canvas").on('mouseleave', function(){
  d3.select('#tooltip')
    .style('opacity', 0.0)
    .style('top', 0 + 'px')
    .style('left',0 + 'px')
    .html('');
})


function createFeedForwardModel() {
    function custom_loss(y_obs, y_pred) {
        t1 = getTemperature1()
        t2 = getTemperature2()
        return tf.mean(bitemperedBinaryLogisticLoss(y_pred, y_obs, t1, t2))
    }

    var model = tf.sequential()
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'randomNormal',
        inputShape: 2,
        activation: 'relu',
        useBias: true
    }))
    model.add(tf.layers.dense({
        units: 5,
        kernelInitializer: 'randomNormal',
        activation: 'relu',
        useBias: true
    }))
    model.add(tf.layers.dense({
        units: 1,
        kernelInitializer: 'randomNormal',
        activation: 'linear',
        useBias: true
    }))
    const optimizer = tf.train.adam(0.1);
    // 
    model.compile({
        optimizer: optimizer,
        loss: custom_loss,
        //shuffle: true
    })
    return model
}

NUM_SHADES = 10

var xScale = d3.scaleLinear()
    .domain([0, 50])
    .range([0, width]);

var yScale = d3.scaleLinear()
    .domain([0, 50])
    .range([0, height]);

let colorScale = d3.scaleLinear()
    .domain([0, 0.25, .5, 0.75, 1])
    .range(["#248ea9", "#82b1bd", "#ffffff", "#ffc7b8", "#ff8b6a"])
    .clamp(true);

let colors = d3.range(0, 1.0001, 1 / NUM_SHADES).map(a => {
    return colorScale(a);
});

color = d3.scaleQuantize()
    .domain([0, 1.0])
    .range(colors);

function generateTestData() {
  var testDataXY = []
  for (var y = 0, p = -1; y < height; ++y) {
      for (var x = 0; x < width; ++x) {
          testDataXY.push(centerCoordinates([x, y]))
      }
  }
  return tf.tensor(testDataXY, [width * height, 2])
}

var testDataTensor = generateTestData();

async function renderDecisionSurface(canvas, model) {
    ctx = canvas.node().getContext("2d");
    ctx.globalAlpha = 0.2;
    var img = ctx.createImageData(width, height);
    t2 = getTemperature2()
    predictedLabels = tf.split(temperedSigmoid(model.predict(testDataTensor), t2, 5), 2, 1)[1].arraySync()
    i = -1;
    for (var y = 0, p = -1; y < height; ++y) {
        for (var x = 0; x < width; ++x) {
            let c = d3.rgb(color(predictedLabels[++i]));
            img.data[++p] = c.r;
            img.data[++p] = c.g;
            img.data[++p] = c.b;
            img.data[++p] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

async function train() {
    var trainDataXY = []
    var trainDataLabel = []
    for (var i = 0; i < trainingDataPoints.length; i++) {
        trainDataXY.push(centerCoordinates([trainingDataPoints[i].x, trainingDataPoints[i].y]))
        trainDataLabel.push(trainingDataPoints[i].label)
    }
    var xDataset = tf.data.array(trainDataXY)
    var yDataset = tf.data.array(trainDataLabel)
    const xyDataset = tf.data.zip(
      {xs: xDataset, ys: yDataset}).batch(1024);
    for (let i = 1; i < 20; ++i) {
        await model.fitDataset(xyDataset, {
            epochs: 10,
            shuffle: true
        })
        updateStatus("$ Training at [" + i * 10 + "/200] epochs.")
        renderDecisionSurface(canvas, model)
    }
    var weights = model.getWeights() 
    var squaredNorm = tf.sum(tf.square(weights[0]))
    for (var i = 1; i < weights.length; ++i) {
        squaredNorm = tf.add(squaredNorm, tf.sum(tf.square(weights[i])))
    }
    // L2(w): " + tf.sqrt(squaredNorm).dataSync()
    updateStatus("$ Done training.")
}

function maybeTrain() {
    if (no_active_points) {
        train()
    } else {
        updateStatus("$ Starting training...")
        setTimeout(maybeTrain, 2000);
    }
}

function resetModel() {
  if (shouldResetModel()) {
    model = createFeedForwardModel();
  }
  train();
}

function duringDragging(d) {
    d3.select(this).attr('cx', d.x = d3.event.x).attr("cy", d.y = d3.event.y);
}

function dragStarted(d) {
    no_active_points = false
    d3.select(this).raise().classed("active", true);
}

async function dragEnd(d) {
    no_active_points = true
    d3.select(this).classed("active", false);
    updateStatus("$ Starting training...")
    setTimeout(maybeTrain, 500);
}