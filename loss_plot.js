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

numLossPlotActivations = 100
var dataset = []
function updateDataset() {
    dataset = d3.range(numLossPlotActivations).map(function(d) {
        x = -5.0 + (10.0 / numLossPlotActivations) * d
        activation = tf.tensor([x], [1])
        probability = tf.tensor([1.0], [1])
        t1 = d3.select("#t1").property("value") / 100.0
        t2 = d3.select("#t2").property("value") / 100.0
        y = tf.mean(bitemperedBinaryLogisticLoss(activation, probability, t1, t2)).arraySync()
        return {
            x: x,
            y: y,
        };
    });
}

var logisticLossDataset = d3.range(numLossPlotActivations).map(function(d) {
    x = -5.0 + (10.0 / numLossPlotActivations) * d
    activation = tf.tensor([x], [1])
    probability = tf.tensor([1.0], [1])
    y = tf.mean(bitemperedBinaryLogisticLoss(activation, probability, 1.0, 1.0)).arraySync()
    return {
        x: x,
        y: y,
    };
});

updateDataset()

function renderLossPlot() {
    var width = 240;
    var height = 160;

    const lossPlotSvg = d3.select('#loss-plot').attr("height", height)
        .attr("width", width);

    const render = data => {
        const title = 'Plot of bi-tempered logistic loss function';
        lossPlotSvg.selectAll("*").remove();

        const xValue = d => d.x;
        const xAxisLabel = 'activations';

        const yValue = d => d.y;
        const yAxisLabel = 'loss';

        const margin = {
            top: 30,
            right: 50,
            bottom: 40,
            left: 50
        };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const xScale = d3.scaleLinear()
            .domain([-5, 5])
            .range([0, innerWidth])
            .nice();

        const yScale = d3.scaleLinear()
            .domain([0, 6])
            .range([innerHeight, 0])
            .nice();

        const g = lossPlotSvg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const xAxis = d3.axisBottom(xScale)
            .tickSize(-innerHeight)
            .tickPadding(5);

        const yAxis = d3.axisLeft(yScale)
            .tickSize(-innerWidth)
            .ticks(7)
            .tickPadding(10);

        const yAxisG = g.append('g').call(yAxis);
        yAxisG.selectAll('.domain').remove();

        yAxisG.append('text')
            .attr('class', 'axis-label')
            .attr('y', -25)
            .attr('x', -innerHeight / 2)
            .attr('fill', 'black')
            .attr('transform', `rotate(-90)`)
            .attr('text-anchor', 'middle')
            .text(yAxisLabel);

        const xAxisG = g.append('g').call(xAxis)
            .attr('transform', `translate(0,${innerHeight})`);

        xAxisG.select('.domain').remove();

        xAxisG.append('text')
            .attr('class', 'axis-label')
            .attr('y', 25)
            .attr('x', innerWidth / 2)
            .attr('fill', 'black')
            .text(xAxisLabel);

        const lineGenerator = d3.line()
            .x(d => xScale(xValue(d)))
            .y(d => yScale(yValue(d)))
            .curve(d3.curveBasis);

        g.append('path')
            .style("stroke", "#248ea9")
            .attr('class', 'line-path')
            .attr('d', lineGenerator(data));

        g.append('path')
            .attr('class', 'line-path')
            .style("stroke", "#ff8b6a")
            .style("stroke-dasharray", "4,4")
            .attr('d', lineGenerator(logisticLossDataset));

        g.append('text')
            .attr('class', 'title')
            .attr('y', -15)
            .text(title);
    };
    render(dataset)
}

renderLossPlot();