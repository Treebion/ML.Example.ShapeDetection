
using Microsoft.OpenApi.Models;
using ShapeDetection.API.Models;

namespace ShapeDetection.API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.
            builder.Services.AddControllers();

            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new OpenApiInfo { Title = "ShapeDetection.API", Version = "v1" });                
            });

            // Add configuration
            builder.Configuration.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);

            // Register Prediction service
            builder.Services.AddSingleton<Prediction>();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "ShapeDetection.API v1"));
            }

            app.UseHttpsRedirection();
            app.UseAuthorization();
            app.MapControllers();
            app.Run();
        }
    }
}
