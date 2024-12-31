//............................. Ray Tracing en C++ con OpenGL .............................
//... @FECORO, 2023 ...
/*
Este es un código de Ray Tracing que utiliza la librería OpenGL para renderizar la escena en una ventana.
-> El código se divide en varias partes, comenzando con la definición de estructuras y clases para los objetos 3D, 
la cámara, las luces y los materiales.

-> La clase principal RayTracer contiene la lógica principal del trazador de rayos, incluyendo la función trace
que calcula la intersección de un rayo con los objetos de la escena, y la función calculateColor que calcula el color

-> La función main crea una instancia de RayTracer, inicializa la escena con objetos y luces, y ejecuta el bucle principal
que maneja la entrada del usuario y renderiza la escena.

-> El código utiliza la biblioteca OpenGL para crear una ventana y renderizar la escena en una textura, que 
se muestra en la ventana.

-> El trazador de rayos implementa reflexiones y refracciones, así como sombras y atenuación de la luz.

-> El código también incluye un sistema de cámara que permite al usuario moverse por la escena con 
las teclas W, A, S y D.

---Tech---:
-> El trazador de rayos utiliza la biblioteca GLAD para cargar las funciones de OpenGL necesarias.
-> El código utiliza la biblioteca glm para operaciones matemáticas y vectores.
-> El trazador de rayos utiliza la biblioteca GLFW para la creación de ventanas y 
el manejo de la entrada del usuario.

---Nota---:
-> Este código es solo un ejemplo básico de un trazador de rayos y puede ser mejorado.

Creado por Felipe Alexander Correa Rodríguez.
*/

//....... stack de librerías:
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <memory>
#include <omp.h> //....... OpenMP para paralelizar el renderizado

//....... estructura para luces
struct Light {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;

    Light(glm::vec3 pos, glm::vec3 col, float intens) 
        : position(pos), color(col), intensity(intens) {}
};

//....... estructura para los materiales, o sea los colores y otras variables.
struct Material {
    glm::vec3 color;
    float reflectivity;
    float specularity;
    float transparency;
    float refractiveIndex;
    
    Material(glm::vec3 c = glm::vec3(1.0f), 
            float r = 0.0f, 
            float s = 0.0f, 
            float t = 0.0f, 
            float ri = 1.0f)
        : color(c), reflectivity(r), specularity(s), 
          transparency(t), refractiveIndex(ri) {}
};

//....... estructura para el resultado de intersección
struct Intersection {
    float distance;
    glm::vec3 point;
    glm::vec3 normal;
    Material material;
    bool hit;

    Intersection() 
        : distance(std::numeric_limits<float>::infinity()), 
          hit(false) {}
};

//....... clase base para objetos 3D
class Object {
protected:
    glm::vec3 position;
    Material material;

public:
    Object(glm::vec3 pos, Material mat) 
        : position(pos), material(mat) {}
    
    virtual float intersect(const glm::vec3& origin, 
                          const glm::vec3& direction) = 0;
    virtual glm::vec3 getNormal(const glm::vec3& point) = 0;
    Material getMaterial() const { return material; }
    virtual ~Object() = default;
};

//....... clase para la cámara
class Camera {
private:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    float yaw, pitch;
    float fov;
    float sensitivity;
    float lastX, lastY;
    bool firstMouse;

public:
    Camera(glm::vec3 pos = glm::vec3(0.0f, 0.0f, 3.0f))
        : position(pos),
          front(glm::vec3(0.0f, 0.0f, -1.0f)),
          up(glm::vec3(0.0f, 1.0f, 0.0f)),
          yaw(-90.0f),
          pitch(0.0f),
          fov(45.0f),
          sensitivity(0.1f),
          firstMouse(true) {
        updateCameraVectors();
    }

    void updateCameraVectors() {
        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(direction);
        right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
        up = glm::normalize(glm::cross(right, front));
    }

    void processMouseMovement(float xpos, float ypos) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;

        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        updateCameraVectors();
    }

    //....... getters y setters
    glm::vec3 getPosition() const { return position; }
    glm::vec3 getFront() const { return front; }
    glm::vec3 getUp() const { return up; }
    glm::vec3 getRight() const { return right; }
    
    void setPosition(const glm::vec3& pos) { position = pos; }
    float getFov() const { return fov; }
};

//....... implementación de objetos 3D específicos
class Sphere : public Object {
private:
    float radius;

public:
    Sphere(glm::vec3 pos, float r, Material mat) 
        : Object(pos, mat), radius(r) {}

    float intersect(const glm::vec3& origin, const glm::vec3& direction) override {
        glm::vec3 oc = origin - position;
        float a = glm::dot(direction, direction);
        float b = 2.0f * glm::dot(oc, direction);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return -1.0f;
        
        float t = (-b - sqrt(discriminant)) / (2.0f * a);
        return t > 0 ? t : (-b + sqrt(discriminant)) / (2.0f * a);
    }

    glm::vec3 getNormal(const glm::vec3& point) override {
        return glm::normalize(point - position);
    }
};

class Plane : public Object {
private:
    glm::vec3 normal;

public:
    Plane(glm::vec3 pos, glm::vec3 norm, Material mat) 
        : Object(pos, mat), normal(glm::normalize(norm)) {}

    float intersect(const glm::vec3& origin, const glm::vec3& direction) override {
        float denom = glm::dot(normal, direction);
        if (abs(denom) > 1e-6) {
            glm::vec3 p0l0 = position - origin;
            float t = glm::dot(p0l0, normal) / denom;
            return t >= 0 ? t : -1;
        }
        return -1;
    }

    glm::vec3 getNormal(const glm::vec3& point) override {
        return normal;
    }
};

//....... shaders
const char* vertexShaderSource = R"(
    #version 410 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
)";

const char* fragmentShaderSource = R"(
    #version 410 core
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D screenTexture;
    void main() {
        FragColor = texture(screenTexture, TexCoord);
    }
)";

//....... clase principal del Ray Tracer
class RayTracer {
private:
    std::vector<std::unique_ptr<Object>> objects;
    std::vector<Light> lights;
    Camera camera;
    int width, height;
    GLuint shaderProgram;
    GLuint VAO, VBO, EBO, texture;
    GLFWwindow* window;
    glm::vec3* frameBuffer;
    const int maxDepth = 5;
    const float epsilon = 0.0001f;

    void createShaderProgram() {
        //....... vertex shader | por si solo
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        checkShaderCompilation(vertexShader, "VERTEX");

        //....... fragment shader | por si solo
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        checkShaderCompilation(fragmentShader, "FRAGMENT");

        //....... shader program | básicamente sombras y fragment juntos
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        checkShaderProgramLinking(shaderProgram);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    void checkShaderCompilation(GLuint shader, const char* type) {
        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            throw std::runtime_error(std::string(type) + " shader compilation failed: " + infoLog);
        }
    }

    void checkShaderProgramLinking(GLuint program) {
        int success;
        char infoLog[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, NULL, infoLog);
            throw std::runtime_error(std::string("Shader program linking failed: ") + infoLog);
        }
    }

    void setupRenderQuad() {
        float vertices[] = {
            //....... positions        //....... texture coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f
        };
        unsigned int indices[] = {
            0, 1, 2,
            0, 2, 3
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    void setupTexture() {
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
        RayTracer* rayTracer = static_cast<RayTracer*>(glfwGetWindowUserPointer(window));
        rayTracer->camera.processMouseMovement(static_cast<float>(xpos), static_cast<float>(ypos));
    }

public:
    RayTracer(int w, int h) : width(w), height(h), camera(glm::vec3(0.0f, 0.0f, 3.0f)) {
        frameBuffer = new glm::vec3[width * height];
    }

    ~RayTracer() {
        delete[] frameBuffer;
        glfwTerminate();
    }

//.................................................| métodos de init
public:
    void init() {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        #ifdef __APPLE__
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif

        window = glfwCreateWindow(width, height, "Ray Tracer", NULL, NULL);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            throw std::runtime_error("Failed to initialize GLAD");
        }

        createShaderProgram();
        setupRenderQuad();
        setupTexture();
    }

    void addObject(Object* obj) {
        objects.emplace_back(obj);
    }

    void addLight(const Light& light) {
        lights.push_back(light);
    }

private:
    Intersection trace(const glm::vec3& origin, const glm::vec3& direction) {
        Intersection closest;
        
        for (const auto& object : objects) {
            float t = object->intersect(origin, direction);
            
            if (t > epsilon && t < closest.distance) {
                closest.distance = t;
                closest.point = origin + direction * t;
                closest.normal = object->getNormal(closest.point);
                closest.material = object->getMaterial();
                closest.hit = true;
            }
        }
        
        return closest;
    }

    glm::vec3 calculateLighting(const Intersection& intersection, const glm::vec3& viewDir) {
        glm::vec3 finalColor(0.0f);
        glm::vec3 ambient = intersection.material.color * 0.1f; //....... Luz ambiental base

        for (const auto& light : lights) {
            glm::vec3 lightDir = glm::normalize(light.position - intersection.point);
            float lightDistance = glm::length(light.position - intersection.point);

            //....... Sombras
            Intersection shadowTest = trace(intersection.point + intersection.normal * epsilon, lightDir);
            if (shadowTest.hit && shadowTest.distance < lightDistance) {
                continue;
            }

            //....... Difuso
            float diff = std::max(glm::dot(intersection.normal, lightDir), 0.0f);
            glm::vec3 diffuse = intersection.material.color * light.color * diff;

            //....... Especular
            glm::vec3 reflectDir = glm::reflect(-lightDir, intersection.normal);
            float spec = pow(std::max(glm::dot(viewDir, reflectDir), 0.0f), 32.0f);
            glm::vec3 specular = light.color * spec * intersection.material.specularity;

            //....... Atenuación
            float attenuation = 1.0f / (1.0f + 0.09f * lightDistance + 0.032f * lightDistance * lightDistance);

            finalColor += (ambient + diffuse + specular) * attenuation * light.intensity;
        }

        return glm::clamp(finalColor, 0.0f, 1.0f);
    }

    glm::vec3 calculateColor(const glm::vec3& origin, const glm::vec3& direction, int depth) {
        if (depth <= 0) return glm::vec3(0.0f);

        Intersection intersection = trace(origin, direction);
        
        if (!intersection.hit) {
            return glm::vec3(0.1f); //....... Color de fondo
        }

        glm::vec3 color = calculateLighting(intersection, -direction);

        //....... Reflexiones
        if (intersection.material.reflectivity > 0.0f) {
            glm::vec3 reflectDir = glm::reflect(direction, intersection.normal);
            glm::vec3 reflectOrigin = intersection.point + intersection.normal * epsilon;
            glm::vec3 reflectColor = calculateColor(reflectOrigin, reflectDir, depth - 1);
            color = glm::mix(color, reflectColor, intersection.material.reflectivity);
        }

        //....... Refracciones
        if (intersection.material.transparency > 0.0f) {
            float ratio = 1.0f / intersection.material.refractiveIndex;
            glm::vec3 refractDir = glm::refract(direction, intersection.normal, ratio);
            glm::vec3 refractOrigin = intersection.point - intersection.normal * epsilon;
            glm::vec3 refractColor = calculateColor(refractOrigin, refractDir, depth - 1);
            color = glm::mix(color, refractColor, intersection.material.transparency);
        }

        return color;
    }

/*
Literalmente lo que hicimos fue recorrer cada pixel de la pantalla, calcular las coordenadas UV correspondientes,
calcular la dirección del rayo, calcular el color del pixel y almacenarlo en el framebuffer. 
Luego, actualizamos la textura con el contenido del framebuffer y renderizamos la textura en la ventana.
*/

//....... .................................................| métodos de render
    void render() {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                //....... convierte a float y calcula las coordenadas UV
                float u = (2.0f * static_cast<float>(x)) / static_cast<float>(width) - 1.0f;
                float v = 1.0f - (2.0f * static_cast<float>(y)) / static_cast<float>(height);
                
                //....... calculamos el aspect ratio y el factor FOV
                float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
                float fovFactor = glm::tan(glm::radians(camera.getFov() * 0.5f));
                
                //....... calcula la dirección del rayo
                glm::vec3 direction = glm::normalize(
                    camera.getFront() +
                    camera.getRight() * (u * fovFactor * aspectRatio) +
                    camera.getUp() * (v * fovFactor)
                );

                //....... calcula el color y lo almacena en el framebuffer
                frameBuffer[y * width + x] = calculateColor(camera.getPosition(), direction, maxDepth);
            }
        }

        //....... actualizar la textura con el contenido del framebuffer
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, frameBuffer);
    }

    void handleInput() {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        float cameraSpeed = 0.05f;
        glm::vec3 pos = camera.getPosition();
        
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            pos += cameraSpeed * camera.getFront();
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            pos -= cameraSpeed * camera.getFront();
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            pos -= camera.getRight() * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            pos += camera.getRight() * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            pos += camera.getUp() * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            pos -= camera.getUp() * cameraSpeed;

        camera.setPosition(pos);
    }

//....... .................................................| método principal
public:
    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            handleInput();
            render();

            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(shaderProgram);
            glBindVertexArray(VAO);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
};

//....... función principal
int main() {
    try {
        RayTracer rayTracer(1280, 720);
        rayTracer.init();

        //....... configuración de la escena
        Material sphereMat(glm::vec3(0.7f, 0.3f, 0.3f), 0.5f, 0.8f, 0.0f);
        Material groundMat(glm::vec3(0.3f, 0.3f, 0.7f), 0.1f, 0.3f, 0.0f);
        Material glassMat(glm::vec3(0.9f), 0.1f, 1.0f, 0.8f, 1.5f);

        //....... añadir objetos
        rayTracer.addObject(new Sphere(glm::vec3(0.0f, 0.0f, -5.0f), 1.0f, sphereMat));
        rayTracer.addObject(new Sphere(glm::vec3(2.0f, 0.0f, -4.0f), 0.7f, glassMat));
        rayTracer.addObject(new Plane(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), groundMat));

        //....... añadir luces
        rayTracer.addLight(Light(glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(1.0f), 1.0f));
        rayTracer.addLight(Light(glm::vec3(-5.0f, 3.0f, -5.0f), glm::vec3(0.5f, 0.5f, 1.0f), 0.8f));

        rayTracer.mainLoop();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

//.................................................| fin del código
// Todos los derechos reservados. @FECORO, 2023.
