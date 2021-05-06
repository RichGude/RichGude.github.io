// Create AM4Charts
am4core.ready(function() {

    // Themes begin
    am4core.useTheme(am4themes_animated);
    // Themes end


    // Create Color Scheme colors for ACF and PACF plots
    var violinColor = am4core.color("#d36161");     // red
    var dataColor = am4core.color("#616cd3");       // blue

    var aluminumColor = am4core.color("#ff8726");   // orange
    var copperColor = am4core.color("#d21a1a");     // red 
    var steelColor = am4core.color("#45d21a");      // green
    var nickelColor = am4core.color("#1c5fe5");     // blue
    var zincColor = am4core.color("#c100e8");       // magenta
    var cpiColor = am4core.color("#7132a8");        // violet

    // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF NORMALIZED FEATURE DATA - XY CHART
        A line chart showing the trend of economic features and aluminum real price from 1990 to 2020. */

    // Create chart instance
    var featureChart = am4core.create("feature_div", am4charts.XYChart);
    // Load price data saved in .json format
    featureChart.dataSource.url = 'EconData/norm_data.json';
    featureChart.dataSource.parser = new am4core.JSONParser();
    featureChart.responsive.enabled = true;

    // Create chart axes
    var dateAxis = featureChart.xAxes.push(new am4charts.DateAxis());
    dateAxis.dataFields.category = "Date";
    dateAxis.title.text = "Year";
    var linValAxis = featureChart.yAxes.push(new am4charts.ValueAxis());
    linValAxis.title.text = "Normalized Feature Value";

    // Create list of 
    var cols = ["D12", "E12", "b/m", "tbl", "AAA", "BAA", "lty", "ntis", "Rfree", "infl", "ltr", "corpr", "svar", "SPvw"];

    // Create series for normalized feature values
    function createFeatureSeries(name) {
        var series = featureChart.series.push(new am4charts.LineSeries());
        series.dataFields.valueY = name;
        series.dataFields.dateX = "Date";
        series.name = name;
        series.strokeWidth = 1.5;
        series.tooltipText = "[bold]{name}[/]: {valueY.formatNumber('#.##')}";

        var segment = series.segments.template;
        segment.interactionsEnabled = true;

        var hoverState = segment.states.create("hover");
        hoverState.properties.strokeWidth = 3;

        var dimmed = segment.states.create("dimmed");
        dimmed.properties.stroke = am4core.color("#dadada");

        //Define hover-over events
        segment.events.on("over", function(event) {
            processOver(event.target.parent.parent.parent);
        });
        segment.events.on("out", function(event) {
            processOut(event.target.parent.parent.parent);
        });
        return series;
    }
    
    // Create a AM Chart series for each feature from current list
    cols.forEach(createFeatureSeries);

    // Create special series for aluminum better highlighting its trends against the other features
    var series = featureChart.series.push(new am4charts.LineSeries());
        series.dataFields.valueY = 'Aluminum';
        series.dataFields.dateX = "Date";
        series.name = 'Aluminum';
        series.strokeWidth = 2;
        series.tooltipText = "[bold]{name}[/]: {valueY.formatNumber('#.##')}";
        series.tooltip.getFillFromObject = false;
        series.tooltip.background.fill = aluminumColor;
        series.stroke = aluminumColor;

        var segment = series.segments.template;
        segment.interactionsEnabled = true;

        var hoverState = segment.states.create("hover");
        hoverState.properties.strokeWidth = 4;

        var dimmed = segment.states.create("dimmed");
        dimmed.properties.stroke = am4core.color("#dadada");

        //Define hover-over events
        segment.events.on("over", function(event) {
            processOver(event.target.parent.parent.parent);
        });
        segment.events.on("out", function(event) {
            processOut(event.target.parent.parent.parent);
        });

    featureChart.legend = new am4charts.Legend();
    featureChart.legend.position = "bottom";
    featureChart.legend.scrollable = true;
    featureChart.legend.itemContainers.template.events.on("over", function(event) {
        processOver(event.target.dataItem.dataContext);
    })

    featureChart.legend.itemContainers.template.events.on("out", function(event) {
        processOut(event.target.dataItem.dataContext);
    })

    // Add cursor
    featureChart.cursor = new am4charts.XYCursor();
    featureChart.cursor.xAxis = dateAxis;

    function processOver(hoveredSeries) {
        hoveredSeries.toFront();

        hoveredSeries.segments.each(function(segment) {
            segment.setState("hover");})

        featureChart.series.each(function(series) {
        if (series != hoveredSeries) {
            series.segments.each(function(segment) {
                segment.setState("dimmed");})
            series.bulletsContainer.setState("dimmed");}
        });
    }

    // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF CORRELATION FACTORS OF FEATURE DATA - HEAT MAP CHART
        A heat map chart showing the correlation of economic features and aluminum real price with each other. */

    var corrChart = am4core.create("corr_div", am4charts.XYChart);
    corrChart.maskBullets = false;
    
    var xAxis = corrChart.xAxes.push(new am4charts.CategoryAxis());
    var yAxis = corrChart.yAxes.push(new am4charts.CategoryAxis());

    // Load corr data saved in .json format
    corrChart.dataSource.url = 'EconData/corr_factors.json';
    corrChart.dataSource.parser = new am4core.JSONParser();
    corrChart.responsive.enabled = true;

    // Create first category axis
    xAxis.dataFields.category = "Fact1";
    xAxis.renderer.grid.template.disabled = true;
    xAxis.renderer.minGridDistance = 20;
    
    // Create second category axis
    yAxis.dataFields.category = "Fact2";
    yAxis.renderer.grid.template.disabled = true;
    yAxis.renderer.inversed = true;
    yAxis.renderer.minGridDistance = 20;

    // Create value series
    var series = corrChart.series.push(new am4charts.ColumnSeries());
    series.dataFields.categoryX = "Fact1";
    series.dataFields.categoryY = "Fact2";
    series.dataFields.value = "Value";
    series.sequencedInterpolation = true;
    series.defaultState.transitionDuration = 3000;

    var columnTemplate = series.columns.template;
    // Set border as background color to create an 'invisible border between cells
    var bgColor = new am4core.InterfaceColorSet().getFor("background");
    columnTemplate.strokeWidth = 1;
    columnTemplate.stroke = bgColor;

    columnTemplate.strokeOpacity = 0.4;
    columnTemplate.tooltipText = "{Fact1}-{Fact2}: [bold]{value.formatNumber('#.####')}[/]";
    columnTemplate.width = am4core.percent(100);
    columnTemplate.height = am4core.percent(100);

    series.heatRules.push({
        target: columnTemplate,
        property: "fill",
        // Set min color as red, and max as red
        min: am4core.color("#ff0000"),
        minValue: -1,
        max: am4core.color("#00ff00"),
        maxValue: 1
    });

    // heat legend
    var heatLegend = corrChart.bottomAxesContainer.createChild(am4charts.HeatLegend);
    heatLegend.width = am4core.percent(100);
    heatLegend.series = series;
    heatLegend.valueAxis.renderer.labels.template.fontSize = 9;
    heatLegend.valueAxis.renderer.minGridDistance = 30;

    // heat legend behavior
    series.columns.template.events.on("over", function(event) {
    handleHover(event.target);
    })

    series.columns.template.events.on("hit", function(event) {
    handleHover(event.target);
    })

    function handleHover(column) {
    if (!isNaN(column.dataItem.value)) {
        heatLegend.valueAxis.showTooltipAt(column.dataItem.value)
    }
    else {
        heatLegend.valueAxis.hideTooltip();
    }
    }

    series.columns.template.events.on("out", function(event) {
    heatLegend.valueAxis.hideTooltip();
    })

    // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF CORRELATION FACTORS OF FEATURE DATA WITH TIME DELAY - HEAT MAP CHART
        A heat map chart showing the correlation of economic features and aluminum real price over time. */

        var aheadCorrChart = am4core.create("ahead_corr_div", am4charts.XYChart);
        aheadCorrChart.maskBullets = false;
        
        var xAxis = aheadCorrChart.xAxes.push(new am4charts.CategoryAxis());
        var yAxis = aheadCorrChart.yAxes.push(new am4charts.CategoryAxis());
    
        // Load lagged corr data saved in .json format
        aheadCorrChart.dataSource.url = 'EconData/ahead_corr.json';
        aheadCorrChart.dataSource.parser = new am4core.JSONParser();
        aheadCorrChart.responsive.enabled = true;
    
        // Create first category axis
        xAxis.dataFields.category = "Fact2";
        xAxis.renderer.grid.template.disabled = true;
        xAxis.renderer.minGridDistance = 20;
        
        // Create second category axis
        yAxis.dataFields.category = "Fact1";
        yAxis.renderer.grid.template.disabled = true;
        yAxis.renderer.inversed = true;
        yAxis.renderer.minGridDistance = 20;
    
        // Create value series
        var series = aheadCorrChart.series.push(new am4charts.ColumnSeries());
        series.dataFields.categoryX = "Fact2";
        series.dataFields.categoryY = "Fact1";
        series.dataFields.value = "Value";
        series.sequencedInterpolation = true;
        series.defaultState.transitionDuration = 3000;
    
        var columnTemplate = series.columns.template;
        // Set border as background color to create an 'invisible border between cells
        var bgColor = new am4core.InterfaceColorSet().getFor("background");
        columnTemplate.strokeWidth = 1;
        columnTemplate.stroke = bgColor;
    
        columnTemplate.strokeOpacity = 0.4;
        columnTemplate.tooltipText = "{Fact2}-{Fact1}: [bold]{value.formatNumber('#.####')}[/]";
        columnTemplate.width = am4core.percent(100);
        columnTemplate.height = am4core.percent(100);
    
        series.heatRules.push({
            target: columnTemplate,
            property: "fill",
            // Set min color as red, and max as red
            min: am4core.color("#ff0000"),
            minValue: -1,
            max: am4core.color("#00ff00"),
            maxValue: 1
        });
    
        // heat legend
        var heatLegend = aheadCorrChart.bottomAxesContainer.createChild(am4charts.HeatLegend);
        heatLegend.width = am4core.percent(100);
        heatLegend.series = series;
        heatLegend.valueAxis.renderer.labels.template.fontSize = 9;
        heatLegend.valueAxis.renderer.minGridDistance = 30;
    
        // heat legend behavior
        series.columns.template.events.on("over", function(event) {
        handleHover(event.target);
        })
    
        series.columns.template.events.on("hit", function(event) {
        handleHover(event.target);
        })
    
        function handleHover(column) {
        if (!isNaN(column.dataItem.value)) {
            heatLegend.valueAxis.showTooltipAt(column.dataItem.value)
        }
        else {
            heatLegend.valueAxis.hideTooltip();
        }
        }
    
        series.columns.template.events.on("out", function(event) {
        heatLegend.valueAxis.hideTooltip();
        })


    // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF SINGLE LAYER WEIGHT FACTOR FEATURE DATA - COLUMN CHART
        A column chart showing the effect of 1-month delayed economic features and aluminum real price on current prices. */

    // Create chart instance
    var weightChart = am4core.create("weight1_div", am4charts.XYChart);
    // Load price data saved in .json format
    weightChart.dataSource.url = 'EconData/alm_sing_weight.json';
    weightChart.dataSource.parser = new am4core.JSONParser();
    weightChart.responsive.enabled = true;

    // Create chart axes
    var categoryAxis = weightChart.xAxes.push(new am4charts.CategoryAxis());
    categoryAxis.dataFields.category = "Factor";
    categoryAxis.title.text = "Economic Factor";
    // Show as many categories as possible
    categoryAxis.renderer.minGridDistance = 10;
    categoryAxis.renderer.labels.template.horizontalCenter = "right";
    categoryAxis.renderer.labels.template.verticalCenter = "middle";
    categoryAxis.renderer.labels.template.rotation = 315;
    var valueAxis = weightChart.yAxes.push(new am4charts.ValueAxis());
    valueAxis.title.text = "Weight Factor Value";

    // Create factor series
    var series = weightChart.series.push(new am4charts.ColumnSeries());
    series.sequencedInterpolation = true;
    series.dataFields.valueY = "Value";
    series.dataFields.categoryX = "Factor";
    series.tooltipText = "{categoryX}:\n[bold]{valueY}[/]";
    series.columns.template.strokeWidth = 0.5;
    series.fillOpacity = 0.8

    series.tooltip.pointerOrientation = "vertical";
    series.columns.template.column.fillOpacity = 0.8;

    // On hover, make corner radiuses bigger
    var hoverState = series.columns.template.column.states.create("hover");
    hoverState.properties.fillOpacity = 1;

    // Color columns separately
    series.columns.template.adapter.add("fill", function(fill, target) {
    return weightChart.colors.getIndex(target.dataItem.index);
    });

    // Cursor
    weightChart.cursor = new am4charts.XYCursor();
})