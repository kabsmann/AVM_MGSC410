# app.R
library(shiny)
library(shinydashboard)
library(xgboost)
library(dplyr)
library(scales)
library(leaflet)
library(DT)
library(tidygeocoder)
library(recipes)

# Load the saved model and preprocessing objects
homes_data <- read.csv('homes_data_final.csv')
xgb_model <- readRDS('xgboost_model.rds')
preprocessing_recipe <- readRDS('preprocessing_recipe.rds')

# Load crime data
crime_data <- read.csv('crime_index_by_city.csv')

# Load income data
income_data <- read.csv('median_income_zipcode.csv')

# Get unique home types
home_type_options <- unique(na.omit(homes_data$homeType))

# UI Definition
ui <- dashboardPage(
  dashboardHeader(title = "California Real Estate Price Predictor"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Price Prediction", tabName = "prediction", icon = icon("dollar-sign")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper { background-color: #f8f9fa; }
        .box { border-top: 3px solid #3c8dbc; }
        .value-box { cursor: pointer; }
        .prediction-value { font-size: 24px; font-weight: bold; color: #3c8dbc; }
        .address-error { color: #dc3545; font-size: 0.875rem; margin-top: 0.25rem; }
        .high-value { color: #28a745; }
        .low-value { color: #dc3545; }
      "))
    ),
    
    tabItems(
      # Prediction Tab
      tabItem(tabName = "prediction",
              fluidRow(
                box(
                  width = 4,
                  title = "Input Parameters",
                  status = "primary",
                  solidHeader = TRUE,
                  
                  # Address inputs
                  textInput("street", "Street Address", placeholder = "123 Main St"),
                  textInput("city", "City", placeholder = "Los Angeles"),
                  textInput("zip", "ZIP Code", placeholder = "90210"),
                  
                  # Property details
                  numericInput("bathrooms", "Bathrooms", value = 1, min = 0, step = 0.5),
                  numericInput("bedrooms", "Bedrooms", value = 1, min = 0),
                  numericInput("lot_size", "Lot Size (sq ft)", value = 500, min = 0),
                  numericInput("property_age", "Property Age (years)", 
                               value = 10, min = 0),
                  numericInput("altitude", "Altitude (ft)", value = 100, min = 0),
                  selectInput("home_type", "Home Type", 
                              choices = home_type_options),
                  numericInput("coffee_shop_count", "Coffee Shop Count", 
                               value = 1, min = 0),
                  numericInput("distance_to_coast", "Distance to Coast (ft)", 
                               value = 5, min = 0),
                  
                  actionButton("predict", "Predict Price", 
                               class = "btn-primary btn-lg btn-block")
                ),
                
                # Results column
                column(width = 8,
                       # Address validation message
                       uiOutput("address_validation"),
                       
                       # Area Statistics
                       fluidRow(
                         box(
                           width = 12,
                           title = "Area Statistics",
                           status = "primary",
                           solidHeader = TRUE,
                           
                           fluidRow(
                             column(6,
                                    h4("Median Income"),
                                    uiOutput("median_income_display")
                             ),
                             column(6,
                                    h4("Crime Index"),
                                    uiOutput("crime_index_display")
                             )
                           )
                         )
                       ),
                       
                       # Prediction Output
                       fluidRow(
                         box(
                           width = 12,
                           title = "Predicted Home Price",
                           status = "primary",
                           solidHeader = TRUE,
                           uiOutput("price_prediction_ui")
                         )
                       ),
                       
                       # Map
                       box(
                         width = NULL,
                         title = "Location Overview",
                         status = "primary",
                         solidHeader = TRUE,
                         leafletOutput("map", height = 400)
                       )
                )
              )
      ),
      
      # About Tab
      tabItem(tabName = "about",
              box(
                width = 12,
                title = "About This Tool",
                status = "info",
                solidHeader = TRUE,
                HTML("
                  <h4>California Real Estate Price Predictor</h4>
                  <p>This tool uses machine learning to predict home prices based on various features:</p>
                  <ul>
                    <li>Property characteristics (bedrooms, bathrooms, lot size)</li>
                    <li>Location features (altitude, distance to coast)</li>
                    <li>Neighborhood characteristics (crime index, coffee shops)</li>
                    <li>Economic indicators (median income)</li>
                  </ul>
                  <p>The predictions are based on an XGBoost model trained on historical housing data.</p>
                ")
              )
      )
    )
  )
)

server <- function(input, output, session) {
  # Validation for California cities (which will be used for ZIP validation too)
  is_valid_ca_city <- reactive({
    req(input$city)
    result <- crime_data %>%
      filter(tolower(city) == tolower(input$city)) %>%
      nrow() > 0
    return(result)
  })
  
  # Validation for ZIP codes in California cities
  is_valid_ca_zip <- reactive({
    req(input$zip)
    # First check if we have a valid California city
    if (!is_valid_ca_city()) {
      return(FALSE)
    }
    # If we have a valid city, just check if the ZIP exists in our income data
    result <- income_data %>%
      filter(zipcode == input$zip) %>%
      nrow() > 0
    return(result)
  })
  
  # Reactive expression for location data
  location_data <- reactive({
    req(input$street, input$city, input$zip)
    
    # Combine address components
    full_address <- paste(
      input$street,
      input$city,
      "CA",
      input$zip,
      "USA"
    )
    
    # Geocode the address
    result <- geo(address = full_address, method = "osm")
    
    if (nrow(result) > 0 && !is.na(result$lat) && !is.na(result$long)) {
      list(
        lat = result$lat,
        lng = result$long,
        valid = TRUE
      )
    } else {
      list(
        valid = FALSE
      )
    }
  })
  
  # Reactive expression for median income
  median_income <- reactive({
    req(input$zip)
    income_data %>% 
      filter(zipcode == input$zip) %>% 
      pull(median_income) %>% 
      head(1)
  })
  
  # Reactive expression for crime index
  crime_index <- reactive({
    req(input$city)
    crime_data %>% 
      filter(tolower(city) == tolower(input$city)) %>% 
      pull(crime_index) %>% 
      head(1)
  })
  
  # Modified median income display with corrected color logic
  output$median_income_display <- renderUI({
    req(input$zip)
    if (!is_valid_ca_zip()) {
      return(
        div(
          style = "background-color: #fee2e2; border: 1px solid #ef4444; padding: 1rem; border-radius: 0.375rem;",
          h3(style = "color: #dc2626; margin: 0; font-size: 1.25rem; font-weight: 600;",
             "Invalid ZIP Code for selected California city")
        )
      )
    }
    income <- median_income()
    class <- if(income >= 100000) "low-value" else "high-value"
    HTML(sprintf(
      "<p class='%s'>$%s</p>",
      class,
      format(income, big.mark = ",")
    ))
  })
  
  # Modified crime index display with corrected color logic
  output$crime_index_display <- renderUI({
    req(input$city)
    if (!is_valid_ca_city()) {
      return(
        div(
          style = "background-color: #fee2e2; border: 1px solid #ef4444; padding: 1rem; border-radius: 0.375rem;",
          h3(style = "color: #dc2626; margin: 0; font-size: 1.25rem; font-weight: 600;",
             "City not found in California crime data")
        )
      )
    }
    crime <- crime_index()
    # Updated color logic for crime index
    class <- if(crime >= 1000) "low-value" else "high-value"
    HTML(sprintf(
      "<p class='%s'>%s</p>",
      class,
      format(crime, big.mark = ",")
    ))
  })
  
  # Address validation output
  output$address_validation <- renderUI({
    loc <- location_data()
    if (!is.null(loc) && !loc$valid) {
      div(class = "address-error",
          "Unable to locate the provided address. Please check and try again.")
    }
  })
  
  # Update map based on geocoded location
  observe({
    loc <- location_data()
    
    if (!is.null(loc) && loc$valid) {
      leafletProxy("map") %>%
        clearMarkers() %>%
        setView(lng = loc$lng, lat = loc$lat, zoom = 15) %>%
        addMarkers(lng = loc$lng, lat = loc$lat,
                   popup = paste(input$street, "<br>",
                                 input$city, "CA",
                                 input$zip))
    }
  })
  
  # Initialize map
  output$map <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%
      setView(lng = -119.4179, lat = 36.7783, zoom = 6)  # Center on California
  })
  
  # Modified price prediction reactive with corrected validation
  price_prediction_reactive <- eventReactive(input$predict, {
    req(input$bathrooms, input$bedrooms, input$lot_size,
        input$property_age, input$altitude,
        input$home_type, input$coffee_shop_count,
        input$distance_to_coast)
    
    # First check if we have a valid California city
    if (!is_valid_ca_city()) {
      return(NULL)
    }
    
    # Then check if we have a valid ZIP code for that city
    if (!is_valid_ca_zip()) {
      return(NULL)
    }

    
    # Get income and crime data
    income <- median_income()
    crime <- crime_index()
    
    # Validate required data
    if (is.null(income) || is.null(crime)) {
      return(NULL)
    }
    
    # Create input dataframe with exact column names matching the recipe
    user_data <- data.frame(
      bathrooms = as.numeric(input$bathrooms),
      bedrooms = as.numeric(input$bedrooms),
      lotSize = as.numeric(input$lot_size),
      median_income = as.numeric(income),
      property_age = as.numeric(input$property_age),
      altitude = as.numeric(input$altitude),
      homeType = as.character(input$home_type),
      coffee_shop_count = as.numeric(input$coffee_shop_count),
      distance_to_coast = as.numeric(input$distance_to_coast),
      crime_index = as.numeric(crime)
    )
    
    # Add error handling for preprocessing
    tryCatch({
      # Preprocess the data
      processed_data <- bake(preprocessing_recipe, new_data = user_data)
      
      # Convert to matrix and make prediction
      pred_matrix <- as.matrix(processed_data)
      prediction <- predict(xgb_model, pred_matrix)
      
      # Return the prediction value
      return(as.numeric(prediction))
    }, error = function(e) {
      # Print error message to console for debugging
      message("Error in prediction: ", e$message)
      return(NULL)
    })
  })
  
  # Modified price prediction UI with corrected validation message
  output$price_prediction_ui <- renderUI({
    if (!is_valid_ca_city()) {
      return(
        div(
          style = "background-color: #fee2e2; border: 1px solid #ef4444; padding: 1rem; border-radius: 0.375rem;",
          h3(style = "color: #dc2626; margin: 0; font-size: 1.25rem; font-weight: 600;",
             "Please enter a valid California city")
        )
      )
    }
    if (!is_valid_ca_zip()) {
      return(
        div(
          style = "background-color: #fee2e2; border: 1px solid #ef4444; padding: 1rem; border-radius: 0.375rem;",
          h3(style = "color: #dc2626; margin: 0; font-size: 1.25rem; font-weight: 600;",
             "Please enter a valid ZIP code for the selected city")
        )
      )
    }
    
    pred <- price_prediction_reactive()
    if (is.null(pred)) {
      HTML("<div class='prediction-value'>Please ensure all inputs are valid</div>")
    } else {
      formatted_price <- format(round(pred), big.mark = ",")
      HTML(sprintf("<div class='prediction-value'>$%s</div>", formatted_price))
    }
  })
  
  # Price per square foot calculation
  output$price_per_sqft <- renderUI({
    pred <- price_prediction_reactive()
    lot_size <- as.numeric(input$lot_size)
    
    if (!is.null(pred) && !is.null(lot_size) && lot_size > 0) {
      price_per_sqft <- pred / lot_size
      formatted_price <- format(round(price_per_sqft), big.mark = ",")
      HTML(sprintf("<div class='prediction-value'>$%s</div>", formatted_price))
    } else {
      HTML("<div class='prediction-value'>-</div>")
    }
  })
}

# Create and run the Shiny app
shinyApp(ui = ui, server = server)