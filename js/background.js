/*
*    background.js
*    Compiling javascript for the background.html file of the Capstone_Viz project
*/

// Create Price Graph
am4core.ready(function() {

    // Themes begin
    am4core.useTheme(am4themes_animated);
    // Themes end

    var aluminumColor = am4core.color("#ff8726"); // orange
    var copperColor = am4core.color("#d21a1a");   // red 
    var steelColor = am4core.color("#45d21a");  // green
    var nickelColor = am4core.color("#1c5fe5");   // blue
    var zincColor = am4core.color("#c100e8");     // purple

    // Create chart instance
    var chart = am4core.create("metal_price_div", am4charts.XYChart);
    // Load price data saved in .json format
    chart.dataSource.url = 'PriceData/priceData.json';
    chart.dataSource.parser = new am4core.JSONParser();
    chart.responsive.enabled = true;

    // Create chart axes
    var dateAxis = chart.xAxes.push(new am4charts.DateAxis());
    dateAxis.dataFields.category = "Date";
    dateAxis.title.text = "Year";
    var linValAxis = chart.yAxes.push(new am4charts.ValueAxis());
    linValAxis.logarithmic = true;
    linValAxis.title.text = "Metal Price (US$/metric ton)";

    // Count number of columns
    var cols = ['Aluminum', 'Copper', 'Iron Ore', 'Nickel', 'Zinc'];
    var colorList = {'Aluminum': aluminumColor, 'Copper': copperColor, 'Iron Ore': steelColor, 'Nickel': nickelColor, 'Zinc': zincColor};

    // Create series
    function createSeries(name) {
        var series = chart.series.push(new am4charts.LineSeries());
        series.dataFields.valueY = name;
        series.dataFields.dateX = "Date";
        series.name = name;   // Need to change name variable for calling purposes...
        // Mouse over tooltip text
        //series.columns.template.tooltipText = "{name} Price in US$ per metric ton (not seasonally adjusted) from years 1990 to 2021";
        series.tooltipText = "[bold]{Date.formatDate('yyyy-MMM')}[/]\n[bold]{name}[/]: {valueY.formatNumber('$#.##')}";
        series.tooltip.getFillFromObject = false;
        series.tooltip.background.fill = colorList[name];
        series.stroke = colorList[name];
        series.strokeWidth = 2;

        var segment = series.segments.template;
        segment.interactionsEnabled = true;

        var hoverState = segment.states.create("hover");
        hoverState.properties.strokeWidth = 4;

        var dimmed = segment.states.create("dimmed");
        dimmed.properties.stroke = am4core.color("#dadada");

        // Define hover-over events
        segment.events.on("over", function(event) {
            processOver(event.target.parent.parent.parent);
        });
        segment.events.on("out", function(event) {
            processOut(event.target.parent.parent.parent);
        });
        return series;
    }
    
    // Create a AM Chart series for each metal
    cols.forEach(createSeries);

    chart.legend = new am4charts.Legend();
    chart.legend.position = "bottom";
    chart.legend.scrollable = true;
    chart.legend.itemContainers.template.events.on("over", function(event) {
        processOver(event.target.dataItem.dataContext);
    })

    chart.legend.itemContainers.template.events.on("out", function(event) {
        processOut(event.target.dataItem.dataContext);
    })

    // Add cursor
    chart.cursor = new am4charts.XYCursor();
    chart.cursor.xAxis = dateAxis;

    function processOver(hoveredSeries) {
        hoveredSeries.toFront();

        hoveredSeries.segments.each(function(segment) {
            segment.setState("hover");})

        chart.series.each(function(series) {
            if (series != hoveredSeries) {
            series.segments.each(function(segment) {
                segment.setState("dimmed");})
            series.bulletsContainer.setState("dimmed");}
        });
    }

    // Add switch for switching between logarithmic scale and linear (for comparing purposes)
    var axisTypeSwitch = chart.legend.createChild(am4core.SwitchButton);
    axisTypeSwitch.leftLabel.text = "log axis";
    axisTypeSwitch.rightLabel.text = "linear"
    axisTypeSwitch.leftLabel.fill = am4core.color("#000");
    axisTypeSwitch.rightLabel.fill = am4core.color("#000");
    
    axisTypeSwitch.events.on("down", function() {
        legendDown = true;
    });
    axisTypeSwitch.events.on("up", function() {
        setTimeout(function() {
        legendDown = false;
        }, 100)
    });
    
    axisTypeSwitch.events.on("toggled", function() {
        // When the switch is toggled and now on, dynamically change the axis to linear scale
        if (axisTypeSwitch.isActive) {
            linValAxis.logarithmic = false;
            cols.show();
        }
        // When the switch is toggled and now off, dynamically change the axis to log scale
        else {
            linValAxis.logarithmic = true;
            cols.show(); 
        }
    })
    



    /* /////////////////////////////
    * Create Geochart Graph *
    */ ////////////////////////////

   // Additional Themes begin
    am4core.useTheme(am4themes_dataviz);
    // Themes end

  var numberFormatter = new am4core.NumberFormatter();

  var backgroundColor = am4core.color("#1e2128");

  // for an easier access by key
  var colors = { Aluminum: aluminumColor, Copper: copperColor, Steel: steelColor, Nickel: nickelColor, Zinc: zincColor };

  var countryColor = am4core.color("#3b3b3b");
  var countryStrokeColor = am4core.color("#000000");
  var buttonStrokeColor = am4core.color("#ffffff");
  var countryHoverColor = am4core.color("#1b1b1b");
  var activeCountryColor = am4core.color("#0f0f0f");

  var currentIndex;
  var currentCountry = "World";

  // last date of the data
  var lastDate = new Date(total_prod[total_prod.length - 1].date);
  var currentDate = lastDate;

  var currentPolygon;

  var countryDataTimeout;

  var currentType = "Aluminum";

  var currentTypeName = "Aluminum";

  var sliderAnimation;

  //////////////////////////////////////////////////////////////////////////////
  // PREPARE DATA
  //////////////////////////////////////////////////////////////////////////////

  // make a map of country indeces for later use
  var countryIndexMap = {};
  // last entry in the country_prod variable chosen to match last data, though not technically necessary
  var list = country_prod[country_prod.length - 1].list;
  for (var i = 0; i < list.length; i++) {
    var country = list[i]
    countryIndexMap[country.id] = i;
  }

  // function that returns current slide
  // if index is not set, get last slide
  function getSlideData(index) {
    if (index == undefined) {
      index = country_prod.length - 1;
    }
    var data = country_prod[index];

    return data;
  }

  // get slide data
  var slideData = getSlideData();

  // as we will be modifying raw data, make a copy
  var mapData = JSON.parse(JSON.stringify(slideData.list));

  var maxU = { Aluminum: 0, Copper: 0, Nickel: 0, Steel: 0, Zinc: 0};

  // Assume the last year will have most production (make sure to parse as integers the json numbers!)
  for (var i = 0; i < mapData.length; i++) {
    var di = mapData[i];
    if (parseInt(di.Aluminum) > maxU.Aluminum) {
        maxU.Aluminum = parseInt(di.Aluminum);
    }
    if (parseInt(di.Copper) > maxU.Copper) {
        maxU.Copper = parseInt(di.Copper);
    }
    if (parseInt(di.Steel) > maxU.Steel) {
        maxU.Steel = parseInt(di.Steel);
    }
    if (parseInt(di.Nickel) > maxU.Nickel) {
        maxU.Nickel = parseInt(di.Nickel);
      }
    if (parseInt(di.Zinc) > maxU.Zinc) {
        maxU.Zinc = parseInt(di.Zinc);
    }
  }

  // END OF DATA

  //////////////////////////////////////////////////////////////////////////////
  // LAYOUT & CHARTS
  //////////////////////////////////////////////////////////////////////////////

  // main container
  // https://www.amcharts.com/docs/v4/concepts/svg-engine/containers/
  var container = am4core.create("geo_prod_div", am4core.Container);
  container.width = am4core.percent(100);
  container.height = am4core.percent(100);

  container.tooltip = new am4core.Tooltip();  
  container.tooltip.background.fill = am4core.color("#000000");
  container.tooltip.background.stroke = aluminumColor;
  container.tooltip.fontSize = "0.9em";
  container.tooltip.getFillFromObject = false;
  container.tooltip.getStrokeFromObject = false;

  // MAP CHART 
  // https://www.amcharts.com/docs/v4/chart-types/map/
  var mapChart = container.createChild(am4maps.MapChart);
  mapChart.height = am4core.percent(80);
  mapChart.zoomControl = new am4maps.ZoomControl();
  mapChart.zoomControl.align = "right";
  mapChart.zoomControl.marginRight = 15;
  mapChart.zoomControl.valign = "middle";
  mapChart.homeGeoPoint = { longitude: 0, latitude: -2 };

  // by default minus button zooms out by one step, but we modify the behavior so when user clicks on minus, the map would fully zoom-out and show world data
  mapChart.zoomControl.minusButton.events.on("hit", showWorld);
  // clicking on a "sea" will also result a full zoom-out
  mapChart.seriesContainer.background.events.on("hit", showWorld);
  mapChart.seriesContainer.background.events.on("over", resetHover);
  mapChart.seriesContainer.background.fillOpacity = 0;
  mapChart.zoomEasing = am4core.ease.sinOut;

  // Show world map in low-resolution rendering for better performance (at a minimal cost of resolution)
  mapChart.geodata = am4geodata_worldLow;

  // Set projection
  // https://www.amcharts.com/docs/v4/chart-types/map/#Setting_projection
  mapChart.projection = new am4maps.projections.Miller();
  mapChart.panBehavior = "move";

  // Map polygon series (defines how country areas look and behave)
  var polygonSeries = mapChart.series.push(new am4maps.MapPolygonSeries());
  polygonSeries.data = mapData;
  polygonSeries.dataFields.id = "id";
  polygonSeries.dataFields.value = "Aluminum";
  polygonSeries.interpolationDuration = 0;

  polygonSeries.exclude = ["AQ"]; // Antarctica is excluded in non-globe projection
  polygonSeries.useGeodata = true;
  polygonSeries.nonScalingStroke = true;
  polygonSeries.strokeWidth = 0.5;
  // this helps to place tooltip in the visual middle of the area
  polygonSeries.calculateVisualCenter = true;
  
  var polygonTemplate = polygonSeries.mapPolygons.template;
  polygonTemplate.fill = countryColor;
  polygonTemplate.fillOpacity = 1
  polygonTemplate.stroke = countryStrokeColor;
  polygonTemplate.strokeOpacity = 0.15
  polygonTemplate.setStateOnChildren = true;
  polygonTemplate.tooltipPosition = "fixed";

  polygonTemplate.events.on("hit", handleCountryHit);
  polygonTemplate.events.on("over", handleCountryOver);
  polygonTemplate.events.on("out", handleCountryOut);

  polygonSeries.heatRules.push({
    "target": polygonTemplate,
    "property": "fill",
    "min": countryColor,    // don't color countries if they have no/little production
    "max": aluminumColor,
    "dataField": "value",
  })

  // you can have pacific - centered map if you set this to -154.8
  mapChart.deltaLongitude = -10;

  // polygon states
  var polygonHoverState = polygonTemplate.states.create("hover");
  polygonHoverState.transitionDuration = 1400;
  polygonHoverState.properties.fill = countryHoverColor;

  var polygonActiveState = polygonTemplate.states.create("active")
  polygonActiveState.properties.fill = activeCountryColor;

  // END OF MAP  

  // top title
  var title = mapChart.titles.create();
  title.fontSize = "1.5em";
  title.text = "World-Wide Metal Commodity Production";
  title.align = "left";
  title.horizontalCenter = "left";
  title.marginLeft = 20;
  title.paddingBottom = 10;
  title.fill = am4core.color("#ffffff");
  title.y = 20;
 
  polygonSeries.heatRules.getIndex(0).max = colors[currentType];
  polygonSeries.heatRules.getIndex(0).minValue = 1;
  polygonSeries.heatRules.getIndex(0).maxValue = maxU[currentType];
  polygonSeries.mapPolygons.template.applyOnClones = true;

  updateCountryTooltip();

  polygonSeries.mapPolygons.each(function(mapPolygon) {
      mapPolygon.fill = mapPolygon.fill;
      mapPolygon.defaultState.properties.fill = undefined;
  })


  // buttons & chart container
  var buttonsAndChartContainer = container.createChild(am4core.Container);
  buttonsAndChartContainer.layout = "vertical";
  buttonsAndChartContainer.height = am4core.percent(45); // make this bigger if you want more space for the chart
  buttonsAndChartContainer.width = am4core.percent(100);
  buttonsAndChartContainer.valign = "bottom";

  // country name and buttons container
  var nameAndButtonsContainer = buttonsAndChartContainer.createChild(am4core.Container)
  nameAndButtonsContainer.width = am4core.percent(100);
  nameAndButtonsContainer.padding(0, 10, 5, 20);
  nameAndButtonsContainer.layout = "horizontal";

  // name of a country and date label
  var countryName = nameAndButtonsContainer.createChild(am4core.Label);
  countryName.fontSize = "1.1em";
  countryName.fill = am4core.color("#ffffff");
  countryName.valign = "middle";

  // buttons container (Aluminum/Copper/Steel/Nickel/Zinc)
  var buttonsContainer = nameAndButtonsContainer.createChild(am4core.Container);
  buttonsContainer.layout = "grid";
  buttonsContainer.width = am4core.percent(100);
  buttonsContainer.x = 10;
  buttonsContainer.contentAlign = "right";

  var chartAndSliderContainer = buttonsAndChartContainer.createChild(am4core.Container);
  chartAndSliderContainer.layout = "vertical";
  chartAndSliderContainer.height = am4core.percent(100);
  chartAndSliderContainer.width = am4core.percent(100);
  chartAndSliderContainer.background = new am4core.RoundedRectangle();
  chartAndSliderContainer.background.fill = am4core.color("#000000");
  chartAndSliderContainer.background.cornerRadius(30, 30, 0, 0)
  chartAndSliderContainer.background.fillOpacity = 0.25;
  chartAndSliderContainer.paddingTop = 12;
  chartAndSliderContainer.paddingBottom = 0;

  // Slider container for line graph
  var sliderContainer = chartAndSliderContainer.createChild(am4core.Container);
  sliderContainer.width = am4core.percent(100);
  sliderContainer.padding(0, 15, 15, 10);
  sliderContainer.layout = "horizontal";

  var slider = sliderContainer.createChild(am4core.Slider);
  slider.width = am4core.percent(100);
  slider.valign = "middle";
  slider.background.opacity = 0.4;
  slider.opacity = 0.7;
  slider.background.fill = am4core.color("#ffffff");
  slider.marginLeft = 20;
  slider.marginRight = 35;
  slider.height = 15;
  slider.start = 1;

  // Initialize map
  //updateMapData(getSlideData(country_prod.length - 1).list);

  // what to do when slider is dragged
  slider.events.on("rangechanged", function(event) {
    var index = Math.round((country_prod.length - 1) * slider.start);
    updateMapData(getSlideData(index).list);
    updateTotals(index);
  })
  // stop animation if dragged
  slider.startGrip.events.on("drag", () => {
    stop();
    if (sliderAnimation) {
      sliderAnimation.setProgress(slider.start);
    }
  });

  // play button
  var playButton = sliderContainer.createChild(am4core.PlayButton);
  playButton.valign = "middle";
  // play button behavior
  playButton.events.on("toggled", function(event) {
    if (event.target.isActive) {
      play();
    } else {
      stop();
    }
  })
  // make slider grip look like play button
  slider.startGrip.background.fill = playButton.background.fill;
  slider.startGrip.background.strokeOpacity = 0;
  slider.startGrip.icon.stroke = am4core.color("#ffffff");
  slider.startGrip.background.states.copyFrom(playButton.background.states)

  // play behavior
  function play() {
    if (!sliderAnimation) {
      sliderAnimation = slider.animate({ property: "start", to: 1, from: 0 }, 50000, am4core.ease.linear).pause();
      sliderAnimation.events.on("animationended", () => {
        playButton.isActive = false;
      })
    }

    if (slider.start >= 1) {
      slider.start = 0;
      sliderAnimation.start();
    }
    sliderAnimation.resume();
    playButton.isActive = true;
  }

  // stop behavior
  function stop() {
    if (sliderAnimation) {
      sliderAnimation.pause();
    }
    playButton.isActive = false;
  }

  // LINE CHART underneath global chart
  var lineChart = chartAndSliderContainer.createChild(am4charts.XYChart);
  lineChart.fontSize = "0.8em";
  lineChart.paddingRight = 30;
  lineChart.paddingLeft = 30;
  lineChart.maskBullets = false;
  lineChart.zoomOutButton.disabled = true;
  lineChart.paddingBottom = 5;
  lineChart.paddingTop = 3;

  // make a copy of data as we will be modifying it
  lineChart.data = JSON.parse(JSON.stringify(total_prod));

  // bottom-aligned date axis
  var dateAxis = lineChart.xAxes.push(new am4charts.DateAxis());
  dateAxis.renderer.minGridDistance = 50;
  dateAxis.renderer.grid.template.stroke = am4core.color("#000000");
  dateAxis.renderer.grid.template.strokeOpacity = 0.25;
  dateAxis.tooltip.label.fontSize = "0.8em";
  dateAxis.tooltip.background.fill = aluminumColor;
  dateAxis.tooltip.background.stroke = aluminumColor;
  dateAxis.renderer.labels.template.fill = am4core.color("#ffffff");
  /*
  dateAxis.renderer.labels.template.adapter.add("fillOpacity", function(fillOpacity, target){
      return dateAxis.valueToPosition(target.dataItem.value) + 0.1;
  })*/

  // right-aligned value axis
  var valueAxis = lineChart.yAxes.push(new am4charts.ValueAxis());
  valueAxis.renderer.opposite = true;
  valueAxis.interpolationDuration = 3000;       // smoothly transition when data changes
  valueAxis.renderer.grid.template.stroke = am4core.color("#000000");
  valueAxis.renderer.grid.template.strokeOpacity = 0.25;
  valueAxis.renderer.minGridDistance = 30;
  valueAxis.renderer.maxLabelPosition = 0.98;
  valueAxis.renderer.baseGrid.disabled = true;
  valueAxis.tooltip.disabled = true;
  valueAxis.extraMax = 0.05;
  valueAxis.maxPrecision = 0;
  valueAxis.renderer.inside = true;
  valueAxis.renderer.labels.template.verticalCenter = "bottom";
  valueAxis.renderer.labels.template.fill = am4core.color("#ffffff");
  valueAxis.renderer.labels.template.padding(2, 2, 2, 2);
  valueAxis.adapter.add("max", function(max, target) {
    if (max < 5) {
      max = 5
    }
    return max;
  })

  valueAxis.adapter.add("min", function(min, target) {
    if (!seriesTypeSwitch.isActive) {
      if (min < 0) {
        min = 0;
      }
    }
    return min;
  })

  // cursor
  lineChart.cursor = new am4charts.XYCursor();
  lineChart.cursor.maxTooltipDistance = 0;
  lineChart.cursor.behavior = "none"; // set zoomX for a zooming possibility
  lineChart.cursor.lineY.disabled = true;
  lineChart.cursor.lineX.stroke = aluminumColor;
  lineChart.cursor.xAxis = dateAxis;
  // this prevents cursor to move to the clicked location while map is dragged
  am4core.getInteraction().body.events.off("down", lineChart.cursor.handleCursorDown, lineChart.cursor)
  am4core.getInteraction().body.events.off("up", lineChart.cursor.handleCursorUp, lineChart.cursor)

  // legend
  lineChart.legend = new am4charts.Legend();
  lineChart.legend.parent = lineChart.plotContainer;
  lineChart.legend.labels.template.fill = am4core.color("#ffffff");
  lineChart.legend.markers.template.height = 8;
  lineChart.legend.contentAlign = "left";
  lineChart.legend.fontSize = "10px";
  lineChart.legend.itemContainers.template.valign = "middle";
  var legendDown = false;
  lineChart.legend.itemContainers.template.events.on("down", function() {
    legendDown = true;
  })
  lineChart.legend.itemContainers.template.events.on("up", function() {
    setTimeout(function() {
      legendDown = false;
    }, 100)
  })


  var seriesTypeSwitch = lineChart.legend.createChild(am4core.SwitchButton);
  seriesTypeSwitch.leftLabel.text = "total prod";
  seriesTypeSwitch.rightLabel.text = "year change"
  seriesTypeSwitch.leftLabel.fill = am4core.color("#ffffff");
  seriesTypeSwitch.rightLabel.fill = am4core.color("#ffffff");

  seriesTypeSwitch.events.on("down", function() {
    legendDown = true;
  })
  seriesTypeSwitch.events.on("up", function() {
    setTimeout(function() {
      legendDown = false;
    }, 100)
  })

  seriesTypeSwitch.events.on("toggled", function() {
    if (seriesTypeSwitch.isActive) {
      if (!columnSeries) {
        createColumnSeries();
      }

      for (var key in columnSeries) {
        columnSeries[key].hide(0);
      }

      for (var key in series) {
        series[key].hiddenInLegend = true;
        series[key].hide();
      }

      columnSeries[currentType].show();
    }
    else {
      for (var key in columnSeries) {
        columnSeries[key].hiddenInLegend = true;
        columnSeries[key].hide();
      }

      for (var key in series) {
        series[key].hiddenInLegend = false;
        series[key].hide();
      }

      series[currentType].show();

    }
  })

  function updateColumnsFill() {
    columnSeries.active.columns.each(function(column) {
      if (column.dataItem.values.valueY.previousChange < 0) {
        column.fillOpacity = 0;
        column.strokeOpacity = 0.6;
      }
      else {
        column.fillOpacity = 0.6;
        column.strokeOpacity = 0;
      }
    })
  }


  // create series
  var aluminumSeries = addSeries("Aluminum", aluminumColor);
  // active series is visible initially
  aluminumSeries.tooltip.disabled = true;
  aluminumSeries.hidden = false;

  var copperSeries = addSeries("Copper", copperColor);
  var steelSeries = addSeries("Steel", steelColor);
  var nickelSeries = addSeries("Nickel", nickelColor);
  var zincSeries = addSeries("Zinc", zincColor);

  var series = { Aluminum: aluminumSeries, Copper: copperSeries, Steel: steelSeries, Nickel: nickelSeries, Zinc: zincSeries };
  // add series
  function addSeries(name, color) {
    var series = lineChart.series.push(new am4charts.LineSeries())
    series.dataFields.valueY = name;
    series.dataFields.dateX = "date";
    series.name = name;
    series.strokeOpacity = 0.6;
    series.stroke = color;
    series.fill = color;
    series.maskBullets = false;
    series.minBulletDistance = 10;
    series.hidden = true;
    series.hideTooltipWhileZooming = true;

    // series bullet
    var bullet = series.bullets.push(new am4charts.CircleBullet());

    // only needed to pass it to circle
    var bulletHoverState = bullet.states.create("hover");
    bullet.setStateOnChildren = true;

    bullet.circle.fillOpacity = 1;
    bullet.circle.fill = backgroundColor;
    bullet.circle.radius = 2;

    var circleHoverState = bullet.circle.states.create("hover");
    circleHoverState.properties.fillOpacity = 1;
    circleHoverState.properties.fill = color;
    circleHoverState.properties.scale = 1.4;

    // tooltip setup
    series.tooltip.pointerOrientation = "down";
    series.tooltip.getStrokeFromObject = true;
    series.tooltip.getFillFromObject = false;
    series.tooltip.background.fillOpacity = 0.2;
    series.tooltip.background.fill = am4core.color("#000000");
    series.tooltip.dy = -4;
    series.tooltip.fontSize = "0.8em";
    series.tooltipText = "Total {name}: {valueY}";

    return series;
  }


  var series = { Aluminum: aluminumSeries, Copper: copperSeries, Steel: steelSeries, Nickel: nickelSeries, Zinc: zincSeries };

  var columnSeries;

  function createColumnSeries() {
    columnSeries = {}
    columnSeries.Aluminum = addColumnSeries("Aluminum", aluminumColor);
    columnSeries.Aluminum.events.on("validated", function() {
      updateColumnsFill();
    })

    columnSeries.Copper = addColumnSeries("Copper", copperColor);
    columnSeries.Steel = addColumnSeries("Steel", steelColor);
    columnSeries.Nickel = addColumnSeries("Nickel", nickelColor);
    columnSeries.Zinc = addColumnSeries("Zinc", zincColor);
  }

  // add series
  function addColumnSeries(name, color) {
    var series = lineChart.series.push(new am4charts.ColumnSeries())
    series.dataFields.valueY = name;
    series.dataFields.valueYShow = "previousChange";
    series.dataFields.dateX = "date";
    series.name = capitalizeFirstLetter(name);
    series.hidden = true;
    series.stroke = color;
    series.fill = color;
    series.columns.template.fillOpacity = 0.6;
    series.columns.template.strokeOpacity = 0;
    series.hideTooltipWhileZooming = true;
    series.clustered = false;
    series.hiddenInLegend = true;
    series.columns.template.width = am4core.percent(50);

    // tooltip setup
    series.tooltip.pointerOrientation = "down";
    series.tooltip.getStrokeFromObject = true;
    series.tooltip.getFillFromObject = false;
    series.tooltip.background.fillOpacity = 0.2;
    series.tooltip.background.fill = am4core.color("#000000");
    series.tooltip.fontSize = "0.8em";
    series.tooltipText = "{name}: {valueY.previousChange.formatNumber('+#,###|#,###|0')}";

    return series;
  }


  lineChart.plotContainer.events.on("up", function() {
    if (!legendDown) {
      slider.start = lineChart.cursor.xPosition * ((dateAxis.max - dateAxis.min) / (lastDate.getTime() - dateAxis.min));
    }
  })


  // data warning label
  var label = lineChart.plotContainer.createChild(am4core.Label);
  label.text = "Metal output production given only on a yearly basis from the USGS National Minerals Information Center:";
  label.fill = am4core.color("#ffffff");
  label.fontSize = "0.8em";
  label.paddingBottom = 4;
  label.opacity = 0.5;
  label.align = "right";
  label.horizontalCenter = "right";
  label.verticalCenter = "bottom";

  // BUTTONS
  // create buttons
  var aluminumButton = addButton("Aluminum", aluminumColor);
  var copperButton = addButton("Copper", copperColor);
  var steelButton = addButton("Steel", steelColor);
  var nickelButton = addButton("Nickel", nickelColor);
  var zincButton = addButton("Zinc", zincColor);

  var buttons = { Aluminum: aluminumButton, Copper: copperButton, Steel: steelButton, Nickel: nickelButton, Zinc: zincButton };

  // add button
  function addButton(name, color) {
    var button = buttonsContainer.createChild(am4core.Button)
    button.label.valign = "middle"
    button.label.fill = am4core.color("#ffffff");
    button.label.fontSize = "11px";
    button.background.cornerRadius(30, 30, 30, 30);
    button.background.strokeOpacity = 0.3
    button.background.fillOpacity = 0;
    button.background.stroke = buttonStrokeColor;
    button.background.padding(2, 3, 2, 3);
    button.states.create("active");
    button.setStateOnChildren = true;

    var activeHoverState = button.background.states.create("hoverActive");
    activeHoverState.properties.fillOpacity = 0;

    var circle = new am4core.Circle();
    circle.radius = 8;
    circle.fillOpacity = 0.3;
    circle.fill = buttonStrokeColor;
    circle.strokeOpacity = 0;
    circle.valign = "middle";
    circle.marginRight = 5;
    button.icon = circle;

    // save name to dummy data for later use
    button.dummyData = name;

    var circleActiveState = circle.states.create("active");
    circleActiveState.properties.fill = color;
    circleActiveState.properties.fillOpacity = 0.5;

    button.events.on("hit", handleButtonClick);

    return button;
  }

  // handle button click
  function handleButtonClick(event) {
    // we saved name to dummy data
    changeDataType(event.target.dummyData);
  }

  // change data type (Aluminum/Copper/Steel/Nickel/Zinc)
  function changeDataType(name) {
    currentType = name;
    currentTypeName = name;

    // make button active
    var activeButton = buttons[name];
    activeButton.isActive = true;
    // make other buttons inactive
    for (var key in buttons) {
      if (buttons[key] != activeButton) {
        buttons[key].isActive = false;
      }
    }
    // tell series new field name
    polygonSeries.dataFields.value = name;

    polygonSeries.dataItems.each(function(dataItem) {
      dataItem.setValue("value", dataItem.dataContext[currentType]);
      dataItem.mapPolygon.defaultState.properties.fill = undefined;
    })

    dateAxis.tooltip.background.fill = colors[name];
    dateAxis.tooltip.background.stroke = colors[name];
    lineChart.cursor.lineX.stroke = colors[name];

    // show series
    if (seriesTypeSwitch.isActive) {
      var activeSeries = columnSeries[name];
      activeSeries.show();
      // hide other series
      for (var key in columnSeries) {
        if (columnSeries[key] != activeSeries) {
          columnSeries[key].hide();
        }
      }
    }
    else {
      var activeSeries = series[name];
      activeSeries.show();
      // hide other series
      for (var key in series) {
        if (series[key] != activeSeries) {
          series[key].hide();
        }
      }
    }

    // update heat rule's maxValue
    polygonSeries.heatRules.getIndex(0).minValue = 1;
    polygonSeries.heatRules.getIndex(0).maxValue = maxU[currentType];
    polygonSeries.heatRules.getIndex(0).max = colors[name];
    updateCountryTooltip();
  }

  // select a country
  function selectCountry(mapPolygon) {
    resetHover();
    polygonSeries.hideTooltip();

    // if the same country is clicked show world
    if (currentPolygon == mapPolygon) {
      currentPolygon.isActive = false;
      currentPolygon = undefined;
      showWorld();
      return;
    }
    // save current polygon
    currentPolygon = mapPolygon;
    var countryIndex = countryIndexMap[mapPolygon.dataItem.id];
    currentCountry = mapPolygon.dataItem.dataContext.name;

    // make others inactive
    polygonSeries.mapPolygons.each(function(polygon) {
      polygon.isActive = false;
    })

    // clear timeout if there is one
    if (countryDataTimeout) {
      clearTimeout(countryDataTimeout);
    }
    // we delay change of data for better performance (so that data is not changed while zooming)
    countryDataTimeout = setTimeout(function() {
      setCountryData(countryIndex);
    }, 1000); // 1000 is one second

    updateTotals(currentIndex);
    updateCountryName();

    mapPolygon.isActive = true;
    mapChart.zoomToMapObject(mapPolygon, getZoomLevel(mapPolygon));
  }

  // change line chart data to the selected countries  
  function setCountryData(countryIndex) {
    // instead of setting whole data array, we modify current raw data so that a nice animation would happen
    for (var i = 0; i < lineChart.data.length; i++) {
      var di = country_prod[i].list;

      var countryData = di[countryIndex];
      var dataContext = lineChart.data[i];
      if (countryData) {
        dataContext.Aluminum = countryData.Aluminum;
        dataContext.Copper = countryData.Copper;
        dataContext.Steel = countryData.Steel;
        dataContext.Nickel = countryData.Nickel;
        dataContext.Zinc = countryData.Zinc;
        valueAxis.min = undefined;
        valueAxis.max = undefined;
      }
      else { // fill in empty countries
        dataContext.Aluminum = 0;
        dataContext.Copper = 0;
        dataContext.Steel = 0;
        dataContext.Nickel = 0;
        dataContext.Zinc = 0;
        valueAxis.min = 0;
        valueAxis.max = 10;
      }
    }

    lineChart.invalidateRawData();
    updateTotals(currentIndex);
    setTimeout(updateSeriesTooltip, 1000);
  }

  function updateSeriesTooltip() {

    var position = dateAxis.dateToPosition(currentDate);
    position = dateAxis.toGlobalPosition(position);
    var x = dateAxis.positionToCoordinate(position);

    lineChart.cursor.triggerMove({ x: x, y: 0 }, "soft", true);
    lineChart.series.each(function(series) {
      if (!series.isHidden) {
        series.tooltip.disabled = false;
        series.showTooltipAtDataItem(series.tooltipDataItem);
      }
    })
  }

  // what happens when a country is rolled-over
  function rollOverCountry(mapPolygon) {

    resetHover();
    if (mapPolygon) {
      mapPolygon.isHover = true;
    }
  }
  // what happens when a country is rolled-out
  function rollOutCountry(mapPolygon) {
    resetHover();
  }

  // calculate zoom level (default is too close)
  function getZoomLevel(mapPolygon) {
    var w = mapPolygon.polygon.bbox.width;
    var h = mapPolygon.polygon.bbox.width;
    // change 2 to smaller walue for a more close zoom
    return Math.min(mapChart.seriesWidth / (w * 2), mapChart.seriesHeight / (h * 2))
  }

  // show world data
  function showWorld() {
    currentCountry = "World";
    currentPolygon = undefined;
    resetHover();

    if (countryDataTimeout) {
      clearTimeout(countryDataTimeout);
    }

    // make all inactive
    polygonSeries.mapPolygons.each(function(polygon) {
      polygon.isActive = false;
    })

    updateCountryName();

    // update line chart data (again, modifying instead of setting new data for a nice animation)
    for (var i = 0; i < lineChart.data.length; i++) {
      var di = total_prod[i];
      var dataContext = lineChart.data[i];

      dataContext.Aluminum = di.Aluminum;
      dataContext.Copper = di.Copper;
      dataContext.Steel = di.Steel;
      dataContext.Nickel = di.Nickel;
      dataContext.Zinc = di.Zinc;
      valueAxis.min = undefined;
      valueAxis.max = undefined;
    }

    lineChart.invalidateRawData();

    updateTotals(currentIndex);
    mapChart.goHome();
  }

  // updates country name and date
  function updateCountryName() {
    countryName.text = currentCountry + ", " + mapChart.dateFormatter.format(currentDate+1, "yyyy");
  }

  // update total values in buttons
  function updateTotals(index) {
    if (!isNaN(index)) {
      var di = total_prod[index];
      var date = new Date(di.date);
      currentDate = date;

      updateCountryName();

      var position = dateAxis.dateToPosition(date);
      position = dateAxis.toGlobalPosition(position);
      var x = dateAxis.positionToCoordinate(position);

      if (lineChart.cursor) {
        lineChart.cursor.triggerMove({ x: x, y: 0 }, "soft", true);
      }
      for (var key in buttons) {
        var count = Number(lineChart.data[index][key])
        if (!isNaN(count)) {
          buttons[key].label.text = capitalizeFirstLetter(key) + ": " + numberFormatter.format(count, '#,###') + " mt";
        }
      }
      currentIndex = index;
    }
  }

  
  // update map data
  function updateMapData(data) {

    maxM = { Aluminum: 0, Copper: 0, Steel: 0, Nickel: 0, Zinc: 0 };

    for (var i = 0; i < data.length; i++) {
      var di = data[i];
        
      if (parseInt(di.Aluminum) > maxM.Aluminum) {
        maxM.Aluminum = parseInt(di.Aluminum);};
      if (parseInt(di.Copper) > maxM.Copper) {
        maxM.Copper = parseInt(di.Copper);};
      if (parseInt(di.Steel) > maxM.Steel) {
        maxM.Steel = parseInt(di.Steel);};
      if (parseInt(di.Nickel) > maxM.Nickel) {
        maxM.Nickel = parseInt(di.Nickel);};
      if (parseInt(di.Zinc) > maxM.Zinc) {
        maxM.Zinc = parseInt(di.Zinc);};
    }
    polygonSeries.heatRules.getIndex(0).minValue = 1;
    polygonSeries.heatRules.getIndex(0).maxValue = maxM[currentType];
    // Smoothly transition from old to new data
    polygonSeries.invalidateRawData();
  }

  // capitalize first letter
  function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }


  function handleImageOver(event) {
    rollOverCountry(polygonSeries.getPolygonById(event.target.dataItem.id));
  }

  function handleImageOut(event) {
    rollOutCountry(polygonSeries.getPolygonById(event.target.dataItem.id));
  }

  function handleImageHit(event) {
    selectCountry(polygonSeries.getPolygonById(event.target.dataItem.id));
  }

  function handleCountryHit(event) {
    selectCountry(event.target);
  }

  function handleCountryOver(event) {
    rollOverCountry(event.target);
  }

  function handleCountryOut(event) {
    rollOutCountry(event.target);
  }

  function resetHover() {
    polygonSeries.mapPolygons.each(function(polygon) {
      polygon.isHover = false;
    })
  }

  container.events.on("layoutvalidated", function() {
    dateAxis.tooltip.hide();
    lineChart.cursor.hide();
    updateTotals(currentIndex);
  });

  // set initial data and names
  updateCountryName();
  changeDataType("Aluminum");
  //populateCountries(slideData.list);

  setTimeout(updateSeriesTooltip, 3000);

  function updateCountryTooltip() {
    polygonSeries.mapPolygons.template.tooltipText = "[bold]{name}: {value.formatNumber('#,###')}[/]\n[font-size:10px]" + currentTypeName + " (metric tons produced)"
  }

  
});

